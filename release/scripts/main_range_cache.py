"""Bounded main-only read-through cache. No network client or implicit download.

The caller supplies the existing counted, range-validating HTTP transport. Every
miss uses its original revision-pinned URL. Blocks share a global LRU budget.
The cache validates strong ETag continuity and the complete Content-Range.
"""
from collections import Counter, OrderedDict
import io
import re
from urllib.parse import urlparse

MIB=1024**2


class RangeCache:
    def __init__(self,http,*,block_bytes=MIB,max_bytes=128*MIB,max_fetch_bytes=8*MIB,max_read_bytes=64*MIB):
        if not 0<block_bytes<=max_fetch_bytes<=max_bytes or max_fetch_bytes%block_bytes:
            raise ValueError("Invalid bounded cache/fetch configuration")
        self.http=http;self.block_bytes=block_bytes;self.max_bytes=max_bytes
        self.max_fetch_bytes=max_fetch_bytes;self.max_read_bytes=max_read_bytes
        self.blocks=OrderedDict();self.resident_bytes=0;self.etags={};self.sizes={};self.failed=set();self.metrics=Counter()

    def register(self,url,size):
        parsed=urlparse(url)
        if parsed.scheme!="https" or parsed.netloc!="huggingface.co" or parsed.query or parsed.fragment or not re.search(r"^/datasets/.+/resolve/[0-9a-f]{40}/.+",parsed.path):
            raise ValueError("Expected original revision-pinned Hugging Face URL")
        if not isinstance(size,int) or size<=0:raise ValueError("Pinned source size required")
        if url in self.sizes and self.sizes[url]!=size:raise ValueError("Pinned size changed")
        if url in self.failed:raise ValueError("Source cache is poisoned after failed integrity check")
        self.sizes[url]=size

    def _reserve(self,needed):
        while self.resident_bytes+needed>self.max_bytes:
            _,value=self.blocks.popitem(last=False);self.resident_bytes-=len(value)
            self.metrics["evicted_blocks"]+=1

    def _put(self,key,data):
        if key in self.blocks:
            old=self.blocks.pop(key);self.resident_bytes-=len(old)
        self._reserve(len(data));self.blocks[key]=data;self.resident_bytes+=len(data)
        self.metrics["peak_resident_bytes"]=max(self.metrics["peak_resident_bytes"],self.resident_bytes)

    def _fetch(self,url,size,start,stop):
        expected=stop-start
        if expected<=0 or expected>self.max_fetch_bytes:raise ValueError("Fetch outside explicit bound")
        # The counted transport remains authoritative, including redirect bodies.
        # Shrinking speculative overfetch is handled by read before this call.
        self._reserve(expected)
        data,headers=self.http.get(url,byte_range=(start,stop-1))
        headers={k.lower():v for k,v in headers.items()}
        if len(data)!=expected or headers.get("content-range")!=f"bytes {start}-{stop-1}/{size}":
            self.failed.add(url)
            raise ValueError("Incomplete or changed pinned range")
        etag=headers.get("etag")
        if not etag or etag.startswith("W/") or not(etag.startswith('"') and etag.endswith('"')):
            self.failed.add(url)
            raise ValueError("Strong source ETag required for cached bytes")
        previous=self.etags.get(url)
        if previous is not None and previous!=etag:
            self.failed.add(url)
            raise ValueError("Source ETag changed within acquisition")
        self.etags[url]=etag
        self.metrics["fetch_requests"]+=1;self.metrics["fetched_body_bytes"]+=len(data)
        self.metrics["largest_fetch_bytes"]=max(self.metrics["largest_fetch_bytes"],len(data))
        return data

    def read(self,url,size,start,count):
        self.register(url,size)
        if start<0 or count<0 or start+count>size or count>self.max_read_bytes:
            raise ValueError("Read outside pinned source or caller-size bound")
        self.metrics["read_calls"]+=1;self.metrics["requested_bytes"]+=count
        if not count:return b""
        end=start+count;cursor=start;output=bytearray(count);block=self.block_bytes
        while cursor<end:
            begin=cursor//block*block;key=(url,begin)
            if key in self.blocks:
                data=self.blocks[key];self.blocks.move_to_end(key)
                stop=min(end,begin+len(data));n=stop-cursor
                output[cursor-start:stop-start]=data[cursor-begin:stop-begin]
                self.metrics["hit_bytes"]+=n;cursor=stop
                del data
                continue
            # Coalesce consecutive missing blocks needed by this read. Read
            # ahead only to the end of the last aligned block, never whole file.
            stop=min(size,begin+block)
            while stop<end and stop-begin+block<=self.max_fetch_bytes and (url,stop) not in self.blocks:
                stop=min(size,stop+block)
            fetch_start,fetch_stop=begin,stop
            remaining=self.http.budget-self.http.received
            if fetch_stop-fetch_start>remaining:
                fetch_start,fetch_stop=cursor,min(stop,end)
                self.metrics["exact_budget_fallbacks"]+=1
            if fetch_stop-fetch_start>remaining:raise ValueError("Remaining transport budget cannot supply required bytes")
            data=self._fetch(url,size,fetch_start,fetch_stop)
            needed_stop=min(end,fetch_stop)
            output[cursor-start:needed_stop-start]=data[cursor-fetch_start:needed_stop-fetch_start]
            self.metrics["miss_requested_bytes"]+=needed_stop-cursor
            # Cache complete blocks only. An exact-budget partial block is used
            # for this read without masquerading as a complete cache entry.
            first=(fetch_start+block-1)//block*block
            for position in range(first,fetch_stop,block):
                block_stop=min(size,position+block)
                if block_stop<=fetch_stop:self._put((url,position),data[position-fetch_start:block_stop-fetch_start])
            cursor=needed_stop
            del data
        return bytes(output)

    def stats(self):
        return {**dict(self.metrics),"resident_bytes":self.resident_bytes,"resident_blocks":len(self.blocks),
            "block_bytes":self.block_bytes,"max_bytes":self.max_bytes,"max_fetch_bytes":self.max_fetch_bytes,
            "max_read_bytes":self.max_read_bytes,"distinct_etags":len(self.etags)}


class CachedRangeFile(io.RawIOBase):
    def __init__(self,cache,url,size):
        super().__init__();cache.register(url,size)
        self.cache,self.url,self.size,self.position=cache,url,size,0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        if whence not in (0,1,2):raise ValueError("Unknown seek mode")
        result=offset+(0 if whence==0 else self.position if whence==1 else self.size)
        if result<0 or result>self.size:raise ValueError("Seek outside pinned source")
        self.position=result;return result
    def read(self,size=-1):
        count=self.size-self.position if size<0 else min(size,self.size-self.position)
        result=self.cache.read(self.url,self.size,self.position,count)
        self.position+=len(result);return result
