from typing import Union
import numpy as np
import sys
import os
import struct
import io
import glob

################################
"""
对于大量二进制文件的读取，如果需要将结果转化成numpy.ndarray，最好的方式是直接使用numpy.frombuffer(推荐)或者numpy.ndarray(buffer)
如果先通过struct,unpack读取tuple再转化成为numpy.ndarray的时间、空间消耗会远远大于上述方法
时间消耗：numpy.frombuffer < numpy.ndarray(buffer) < struct.unpack < np.array(struct.unpack())
参考链接：https://stackoverflow.com/questions/54679949/unpacking-binary-file-using-struct-unpack-vs-np-frombuffer-vs-np-ndarray-vs-np-f/54681417#54681417?newreg=320f8a5066014cbd925bdc007897b523
""" 
################################
class Read():
    def __init__(self, idLen=8):
        self.idLen = idLen # id length in bytes for snapshots
        self.deBugFlag = False
        self.NDFIELD_MAX_DIMS = 20

    def setPath(self, dirname, filename):
        self.name = filename
        self.dirpath = dirname
        self.filename = self.dirpath + "/" + filename

    def ReadGadget(self, type, target, FlagPtype=np.array([False, True, False, False, False, False]), deBugFlag=True, nFiles=-1, MpcUnit=1000):
        self.deBugFlag = deBugFlag
        # totalLen = len(os.listdir(self.dirpath))
        totalLen = len(glob.glob(self.filename)) if nFiles == 1 else len(glob.glob(self.filename + ".*"))
        # print(totalLen)
        if self.deBugFlag == True:
            print("Total %s files in %s"%(totalLen, self.dirpath))
        nFiles = totalLen if nFiles < 0 else nFiles
        s = {}
        start = 0
        length = 0
        scale = MpcUnit / 1000.

        for fid in range(nFiles):
            if totalLen == 1:
                filename = self.filename
            else:
                # filename = self.filename + "." + str(fid)
                filename = "%s.%s"%(self.filename, fid)
            with open(filename, "rb") as f:
                if self.deBugFlag:
                    print("--------------Opening %s.%s --------------"%(self.name, fid))
                    print("--------------Reading Header--------------")
                s["header"] = self.ReadGadgetHeader(f, type)
                Nall = np.sum(s["header"]['Nall'][FlagPtype]) if type=="snapshot" else s["header"]['NgroupsTot']
                if fid == 0:
                    s = self.__genDict__(s, Nall, type)
                if type == "snapshot":
                    length = s["header"]["Npart"] * FlagPtype
                    Massarr = s["header"]['Massarr']
                    FlagMass = (Massarr > 0) & FlagPtype
                    for item in ["pos", "vel", "id"]:
                        s = self.__ReadBody__(s, f, type, item, FlagPtype, FlagMass, start, length, scale)
                    start = start + np.sum(length)
                elif type == "groups":
                    length = s["header"]["Ngroups"]
                    for item in ["GroupLen", "GroupMass", "GroupPos", "", "GroupID"]:
                        s = self.__ReadBody__(s, f, type, item, None, None, start, length, scale)
                    start = start + length
                if self.deBugFlag:
                    print("--------------Closing %s.%s --------------"%(self.name, fid))
            
            # Check total length
            if fid == totalLen - 1:
                for key, value in s.items():
                    if key != "header" and len(value) != Nall and len(value) != 0:
                        raise ValueError("%s length %s does not match Nall %s"%(key, len(value), Nall))
        dict = {}
        dict["header"] = s["header"]
        for item in target:
            if not s.__contains__(item):
                print("Warning: target %s is not supported"%(item))
                continue
            dict[item] = s[item]
        return dict
    
    def ReadNDskel(self, target="All", deBugFlag=False, MpcUnit=1000, Nodestr=False, Segstr=False):
        self.nsegdata = 0
        self.nnodedata = 0
        self.nsegs = 0
        self.nnodes = 0
        self.ndims = 0
        self.deBugFlag = deBugFlag
        s = {}
        scale = MpcUnit / 1000.
        filename = self.filename
        with open(filename, "rb") as f:
            if self.deBugFlag:
                print("--------------Opening %s --------------"%(self.filename))
            s["filetype"] = self.ReadNDskelTag(f)
            s["header"] = self.ReadNDskelHeader(f)
            s["seg_data_info"] = self.ReadNDskelSegDataInfo(f)
            s["node_data_info"] = self.ReadNDskelNodeDataInfo(f)
            s["segpos"] = self.ReadNDskelSegPos(f)
            s["nodepos"] = self.ReadNDskelNodePos(f, length=self.nnodes*self.ndims*4)
            s["segdata"] = self.ReadNDskelSegData(f)
            s["nodedata"] = self.ReadNDskelNodeData(f)
            if Nodestr:
                s["nodestr"] = self.ReadNDskelNodeStr(f)
            if Segstr:
                s["segstr"] = self.ReadNDskelSegStr(f)
            if self.deBugFlag:
                print("Read seg_str and node_str are not implemented yet")
            
            if self.deBugFlag:
                print("--------------Closing %s --------------"%(self.filename))
        
        dict = {}
        dict["header"] = s["header"]
        if target == "All":
            return s
        else:
            for item in target:
                if not s.__contains__(item):
                    print("Warning: target %s is not supported"%(item))
                    continue
                dict[item] = s[item]
            return dict
        
    def ReadNDfield(self, target="All", deBugFlag=False):
        s = {}
        self.deBugFlag = deBugFlag
        filename = self.filename
        with open(filename, "rb") as f:
            if self.deBugFlag:
                print("--------------Opening %s --------------"%(self.filename))
            s["tag"] = self.ReadNDskelTag(f)
            s["header"] = self.ReadNDfieldHeader(f, length=652)
            
            if self.deBugFlag:
                print("ndims: %s"%(s["header"]["ndims"]))
                print("dims: %s"%(s["header"]["dims"]))
                print("fdims_index: %s"%(s["header"]["fdims_index"]))
                
            tmp = 1
            if (s["header"]["dims"][0] == s["header"]["ndims"]) or (s["header"]["fdims_index"] == 1):
                print("Reading particle coordinates")
                tmp = s["header"]["dims"][0] * s["header"]["dims"][1]
                self.type = "coord"
                size = s["header"]["dims"][1::-1]
            else:
                print("Reading grid values")
                for i in range(s["header"]["ndims"]):
                    tmp *= s["header"]["dims"][i]
                self.type = "grid"
                size = s["header"]["dims"][0:s["header"]["ndims"]]
            s["data"] = self.ReadNDfieldData(f, shape=size, dtype=s["header"]["datatype"])
            
            if self.deBugFlag:
                print("--------------Closing %s --------------"%(self.filename))
        return s
    
    def WriteNDfield(self, filename, ndims, dims, fdims_index, datatype, x0, delta, data:np.ndarray, order='C', comment=None, deBugFlag=False):
        self.deBugFlag = deBugFlag
        comment = "Created by %s"%(self.__class__.__name__) if comment is None else comment
        print("--------------Writing %s --------------"%(filename))
        with open(filename, "wb") as f:
            # Write tag
            print("--------------Writing tag--------------") if self.deBugFlag else None
            self.__WriteBlock__(f, "NDFIELD", length=16, order=order)
            # Write header
            print("--------------Writing header--------------") if self.deBugFlag else None
            s = {}
            s["comment"] = struct.pack("80s", comment.encode())
            s["ndims"] = np.int32(ndims).tobytes()
            if len(dims) < self.NDFIELD_MAX_DIMS:
                dims = np.append(dims, np.zeros(20 - len(dims), dtype=np.int32))
            s["dims"] = dims.astype(np.int32)
            s["fdims_index"] = np.int32(fdims_index).tobytes()
            DATATYPE = (np.char, "not supported", np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)
            tmp = 1 << 0
            for i in range(10):
                if DATATYPE[i] == datatype:
                    s["datatype"] = np.int32(tmp).tobytes()
                    break
                tmp = tmp << 1
            if len(x0) < self.NDFIELD_MAX_DIMS:
                x0 = np.append(x0, np.zeros(20 - len(x0), dtype=np.float32))
            s["x0"] = x0
            if len(delta) < self.NDFIELD_MAX_DIMS:
                delta = np.append(delta, np.zeros(20 - len(delta), dtype=np.float32))
            s["delta"] = delta
            s["dummy"] = struct.pack('160s', b'')
            self.__WriteBlock__(f, s, length=652, order=order)
            # Write data
            print("--------------Writing data--------------") if self.deBugFlag else None
            length = np.prod(data.shape) * np.dtype(datatype).itemsize
            self.__WriteBlock__(f, data, length=length, order=order)
        print("--------------Closing %s --------------"%(filename))

    def ReadGadgetHeader(self, file:io.BufferedReader, type:str):
        dict = {}
        binary, _ = self.__ReadBlock__(file)
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        offset = 0
        if type == "snapshot":
            # header = struct.unpack('6I6d2d2i6i2i4d2i6ii60s', binary) # 256 bytes in total
            dict["Npart"], offset = self.__BufferUnpack__(binary, count=6, dtype=np.uint32, size=6*4, offset=offset)
            dict["Massarr"], offset = self.__BufferUnpack__(binary, count=6, dtype=np.float64, size=6*8, offset=offset)
            dict["Time"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
            dict["Redshift"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
            dict["FlagSfr"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int32, size=1*4, offset=offset)
            dict["FlagFeedback"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int32, size=1*4, offset=offset)
            dict["Nall"], offset = self.__BufferUnpack__(binary, count=6, dtype=np.int32, size=6*4, offset=offset)
            dict["FlagCooling"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int32, size=1*4, offset=offset)
            dict["NumFiles"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int32, size=1*4, offset=offset)
            dict["BoxSize"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
            dict["Omega0"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
            dict["OmegaLambda"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
            dict["HubbleParam"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
            dict["FlagAge"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int32, size=1*4, offset=offset)
            dict["FlagMetals"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int32, size=1*4, offset=offset)
            dict["NallHW"], offset = self.__BufferUnpack__(binary, count=6, dtype=np.int32, size=6*4, offset=offset)
            dict["flag_entr_ics"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int32, size=1*4, offset=offset)
            #----------------------------------------------------#
            # If last 60 Bytes are used, add more elements here 
            self.__BufferUnpack__(binary, count=60, dtype="S1", size=60*1, offset=offset)
            #----------------------------------------------------#
        elif type == "groups":
            # header = struct.unpack('6qi3d', binary) # 80 Bytes in total
            dict["Ngroups"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int64, size=1*8, offset=offset)
            dict["Nsubhalos"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int64, size=1*8, offset=offset)
            dict["Nids"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int64, size=1*8, offset=offset)
            dict["NgroupsTot"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int64, size=1*8, offset=offset)
            dict["NsubhalosTot"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int64, size=1*8, offset=offset)
            dict["NidsTot"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int64, size=1*8, offset=offset)
            dict["NumFiles"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.int32, size=1*4, offset=offset)
            dict["Time / a"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
            dict["Redshift"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
            dict["BoxSize"], offset = self.__BufferUnpack__(binary, count=1, dtype=np.float64, size=1*8, offset=offset)
        return dict
    
    def ReadNDskelTag(self, file:io.BufferedReader, length:int=0):
        print("------Reading NDskel Tag------") if self.deBugFlag else None
        binary, _ = self.__ReadBlock__(file)
        if self.deBugFlag:
            print("Reading NDskel Tag")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        tag, _ = self.__BufferUnpack__(binary, 1, 'S16', 1*16, 0)
        print("Reading NDskel Tag finished!") if self.deBugFlag else None
        return tag.decode('utf-8')
    
    def ReadNDskelHeader(self, file:io.BufferedReader, length:int=500):
        # Warning: The length of header is 500 Bytes, while the dummy bytes of Fortran shows a wrong length
        print("------Reading NDskel Header------") if self.deBugFlag else None
        dict = {}
        binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDskel Header")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        offset = 0
        dict["comment"], offset = self.__BufferUnpack__(binary, 1, 'S80', 1*80, offset)
        dict["comment"] = dict["comment"].decode('utf-8')
        dict["ndims"], offset = self.__BufferUnpack__(binary, 1, np.int32, 1*4, offset)
        dict["dims"], offset = self.__BufferUnpack__(binary, 20, np.int32, 20*4, offset)
        self.ndims = dict["ndims"]
        dict["x0"], offset = self.__BufferUnpack__(binary, 20, np.float64, 20*8, offset)
        dict["delta"], offset = self.__BufferUnpack__(binary, 20, np.float64, 20*8, offset)
        dict["nsegs"], offset = self.__BufferUnpack__(binary, 1, np.int32, 1*4, offset)
        self.nsegs = dict["nsegs"]
        dict["nnodes"], offset = self.__BufferUnpack__(binary, 1, np.int32, 1*4, offset)
        self.nnodes = dict["nnodes"]
        dict["nsegdata"], offset = self.__BufferUnpack__(binary, 1, np.int32, 1*4, offset)
        self.nsegdata = dict["nsegdata"]
        dict["nnodedata"], offset = self.__BufferUnpack__(binary, 1, np.int32, 1*4, offset)
        self.nnodedata = dict["nnodedata"]
        print("ndims: %s"%(dict["ndims"]), "nsegs: %s"%(dict["nsegs"]), "nnodes: %s"%(dict["nnodes"])) if self.deBugFlag else None
        print("Reading NDskel Header finished!") if self.deBugFlag else None
        return dict
    
    def ReadNDfieldHeader(self, file:io.BufferedReader, length:int=652):
        dict = {}
        binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDfield Header")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        offset = 0
        dict["comment"], offset = self.__BufferUnpack__(binary, 1, 'S80', 1*80, offset)
        dict["comment"] = dict["comment"].decode('utf-8')
        dict["ndims"], offset = self.__BufferUnpack__(binary, 1, np.int32, 1*4, offset)
        dict["dims"], offset = self.__BufferUnpack__(binary, 20, np.int32, 20*4, offset)
        self.ndims = dict["ndims"]
        dict["fdims_index"], offset = self.__BufferUnpack__(binary, 1, np.int32, 1*4, offset)
        tmp, offset = self.__BufferUnpack__(binary, 1, np.int32, 1*4, offset)
        datatype = (np.char, "not supported", np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)
        for i in range(10):
            if tmp & (1 << i):
                dict["datatype"] = datatype[i]
                break
        dict["x0"], offset = self.__BufferUnpack__(binary, 20, np.float64, 20*8, offset)
        dict["delta"], offset = self.__BufferUnpack__(binary, 20, np.float64, 20*8, offset)
        _, _ = self.__BufferUnpack__(binary, 1, 'S160', 1*160, offset)
        return dict
    
    def ReadNDskelSegDataInfo(self, file:io.BufferedReader, length:int=0):
        print("------Reading NDskel Seg Data Info------") if self.deBugFlag else None
        binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDskel Seg Data Info")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        if (len(binary) / 20) != self.nsegdata:
            raise ValueError(f"length of binary {len(binary)} is not equal to {self.nsegdata*20}")
        res, _ = self.__BufferUnpack__(binary, self.nsegdata, 'S20', self.nsegdata*20, 0)
        res = [comment.decode('utf-8') for comment in res]
        print("Reading NDskel Seg Data Info finished!") if self.deBugFlag else None
        return res
    
    def ReadNDskelNodeDataInfo(self, file:io.BufferedReader, length:int=0):
        print("------Reading NDskel Node Data Info------") if self.deBugFlag else None
        binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDskel Node Data Info")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        if (len(binary) / 20) != self.nnodedata:
            raise ValueError(f"length of binary {len(binary)} is not equal to {self.nnodedata*20}")
        res, _ = self.__BufferUnpack__(binary, self.nnodedata, 'S20', self.nnodedata*20, 0)
        res = [comment.decode('utf-8') for comment in res]
        print("Reading NDskel Node Data Info finished!") if self.deBugFlag else None
        return res
    
    def ReadNDskelSegPos(self, file:io.BufferedReader, length:int=0):
        # Note: the size of segpos is nsegs * 2 * ndims, one can find the segpos of segindex[j] by segpos[segindex[j]*2*ndims:(segindex[j]+1)*2*ndims]
        # Warning: the shape of segpos is not the same as in tutorial, it is (nsegs, 2, ndims)
        print("------Reading NDskel Seg Pos------") if self.deBugFlag else None
        if length <= 0:
            binary, nBytes = self.__ReadBlock__(file, length=4*2*self.nsegs*self.ndims)
            print(f"WARNING: Reading NDskel Seg Pos with block length {nBytes} failed, try reading with length {4*2*self.nsegs*self.ndims}") if nBytes != 4*2*self.nsegs*self.ndims else None
        else:
            binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDskel Seg Pos")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        if (len(binary) / 4) != self.nsegs * 2 * self.ndims:
            raise ValueError(f"length of binary {len(binary)} is not equal to (nsegs,2,ndims) {self.nsegs} * 2 * {self.ndims}")
        res, _ = self.__BufferUnpack__(binary, self.nsegs * 2 * self.ndims, np.float32, self.nsegs * 2 * self.ndims * 4, 0)
        res = res.reshape(self.nsegs, 2, self.ndims)
        print("Reading NDskel Seg Pos finished!") if self.deBugFlag else None
        return res
    
    def ReadNDskelNodePos(self, file:io.BufferedReader, length:int=0):
        # Note: the size of segpos is nnodes * ndims, one can find the segpos of segindex[j] by segpos[segindex[j]*2*ndims:(segindex[j]+1)*2*ndims]
        # Warning: The length of NodePos is nnodes * ndims * 4 Bytes, while the dummy bytes of Fortran doubkes the correct length
        print("------Reading NDskel Node Pos------") if self.deBugFlag else None
        if length <= 0:
            binary, nBytes = self.__ReadBlock__(file, length=4*self.nnodes*self.ndims)
            print(f"WARNING: Reading NDskel Node Pos with block length {nBytes} failed, try reading with length {4*self.nnodes*self.ndims}") if nBytes != 4*self.nnodes*self.ndims else None
        else:
            binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDskel Node Pos")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        if (len(binary) / 4) != self.nnodes * self.ndims:
            raise ValueError(f"length of binary {len(binary)} is not equal to (nnodes,ndims) {self.nnodes} * {self.ndims}")
        res, _ = self.__BufferUnpack__(binary, self.nnodes * self.ndims, np.float32, self.nnodes * self.ndims * 4, 0)
        res = res.reshape(self.nnodes, self.ndims)
        print("Reading NDskel Node Pos finished!") if self.deBugFlag else None
        return res
    
    def ReadNDskelSegData(self, file:io.BufferedReader, length:int=0):
        print("------Reading NDskel Seg Data------") if self.deBugFlag else None
        if length <= 0:
            binary, nBytes = self.__ReadBlock__(file, length=8*self.nsegs*self.nsegdata)
            print(f"WARNING: Reading NDskel Seg Data with block length {nBytes} failed, try reading with length {8*self.nsegs*self.nsegdata}") if nBytes != 8*self.nsegs*self.nsegdata else None
        else:
            binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDskel Seg Data")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        if (len(binary) / 8) != self.nsegs * self.nsegdata:
            raise ValueError(f"length of binary {len(binary)} is not equal to (nsegss,nsegdata) {self.nsegs} * {self.nsegdata}")
        res, _ = self.__BufferUnpack__(binary, self.nsegs * self.nsegdata, np.float64, self.nsegs * self.nsegdata * 8, 0)
        res = res.reshape(self.nsegs, self.nsegdata)
        print("Reading NDskel Seg Data finished!") if self.deBugFlag else None
        return res
    
    def ReadNDskelNodeData(self, file:io.BufferedReader, length:int=0):
        print("------Reading NDskel Node Data------") if self.deBugFlag else None
        if length <= 0:
            binary, nBytes = self.__ReadBlock__(file, length=8*self.nnodes*self.nnodedata)
            print(f"WARNING: Reading NDskel Node Data with block length {nBytes} failed, try reading with length {8*self.nnodes*self.nnodedata}") if nBytes != 8*self.nnodes*self.nnodedata else None
        else:
            binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDskel Node Data")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        if (len(binary) / 8) != self.nnodes * self.nnodedata:
            raise ValueError(f"length of binary {len(binary)} is not equal to (nnodes,nnodedata) {self.nnodes} * {self.nnodedata}")
        res, _ = self.__BufferUnpack__(binary, self.nnodes * self.nnodedata, np.float64, self.nnodes * self.nnodedata * 8, 0)
        res = res.reshape(self.nnodes, self.nnodedata)
        print("Reading NDskel Node Data finished!") if self.deBugFlag else None
        return res
    
    def ReadNDfieldData(self, file:io.BufferedReader, shape, dtype):
        length = np.prod(shape) * np.dtype(dtype).itemsize
        binary, _ = self.__ReadBlock__(file, length)
        if self.deBugFlag:
            print("Reading NDfield Data")
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        if self.type == "coord":
            res, _ = self.__BufferUnpack__(binary, np.prod(shape), dtype, length, 0)
            res = res.reshape(shape)
        elif self.type == "grid":
            res, _ = self.__BufferUnpack__(binary, np.prod(shape), dtype, length, 0)
            res = res.reshape(shape)
        else:
            raise ValueError(f"Unknown type {self.type}")
        return res
    
    def ReadNDskelNodeStr(self, file:io.BufferedReader, length:int=0):
        # Note: Begins from Line 1539 of ${DISPERSE_SRC}/src/C/NDskeleton.c
        # The dummy bytes of Fortran is 0 instead of correct size.
        # All nnodes are contained in a block.
        # TODO: Optimize large file reading
        if self.deBugFlag:
            print("Reading NDskel Node Str")
        print("Warning: dummy block is not 0, probably some erroe occurs!") if ReadBuffer(file, 4, np.int32) != 0 else None
        NodeStr_List = []
        for i in range(self.nnodes):
            tmp = NDskl_node_str()
            tmp.pos_index = ReadBuffer(file, 4, np.int32)[0]
            tmp.flags = ReadBuffer(file, 4, np.int32)[0]
            tmp.nnext = ReadBuffer(file, 4, np.int32)[0]
            tmp.type = ReadBuffer(file, 4, np.int32)[0]
            tmp.index = ReadBuffer(file, 4, np.int32)[0]
            tmp.nsegs = ReadBuffer(file, tmp.nnext * 4, np.int32)
            array = ReadBuffer(file, tmp.nnext * 2 * 4, np.int32).reshape(tmp.nnext, 2)
            tmp.nextNode = array[:,0]
            tmp.nextSeg = array[:,1]
            tmp.__checkValid__()
            NodeStr_List.append(tmp)
        print("Warning: check block is not 0, probably some erroe occurs!") if ReadBuffer(file, 4, np.int32) != 0 else None
        if self.deBugFlag:
            print("Reading NDskel Node Str finished!")
        return NodeStr_List
    
    def ReadNDskelSegStr(self, file:io.BufferedReader, length:int=0):
        # Note: Begins from Line 1564 of ${DISPERSE_SRC}/src/C/NDskeleton.c
        # The dummy bytes of Fortran is 0 instead of correct size.
        # All nnodes are contained in a block.
        # TODO: Optimize large file reading
        if self.deBugFlag:
            print("Reading NDskel Seg Str")
        print("Warning: dummy block is not 0, probably some erroe occurs!") if ReadBuffer(file, 4, np.int32) != 0 else None
        SegStr_List = []
        for i in range(self.nsegs):
            tmp = NDskl_seg_str()
            # Old version
            tmp.pos_index = ReadBuffer(file, 4, np.int32)[0]
            tmp.nodes = ReadBuffer(file, 2*4, np.int32)
            tmp.flags = ReadBuffer(file, 4, np.int32)[0]
            tmp.index = ReadBuffer(file, 4, np.int32)[0]
            tmp.next_seg = ReadBuffer(file, 4, np.int32)[0]
            tmp.prev_seg = ReadBuffer(file, 4, np.int32)[0]
            tmp.__checkValid__()
            SegStr_List.append(tmp)
        print("Warning: dummy block is not 0, probably some erroe occurs!") if ReadBuffer(file, 4, np.int32) != 0 else None
        if self.deBugFlag:
            print("Reading NDskel Seg Str finished!")
        return SegStr_List
    
    def __ReadBody__(self, s:dict, file:io.BufferedReader, type:str, target:str, FlagPtype, FlagMass, start, length, scale):
        if self.deBugFlag:
            print("Reading item", target)
        endFlag = 1
        if type == "snapshot":
            s, endFlag = self.__SnapBinaryUnpack__(s, file, FlagPtype, FlagMass, target, start, length, scale)
        elif type == "groups":
            s, endFlag = self.__GroupBinaryUnpack__(s, file, target, start, length, scale)
        else:
            print("Read type not supported")
        if endFlag == 1:
            print("Reach the end of the file, target %s is not contained in %s"%(target, type))
        return s
    
    def __genDict__(self, s, Nall:int, type)->dict:
        if type == "snapshot":
            s['pos'] = np.zeros([Nall, 3], dtype=np.float32)
            s['vel'] = np.zeros([Nall, 3], dtype=np.float32)
            s['id'] = np.zeros([Nall, 1], dtype=np.uint64)
            s['mass'] = np.zeros([Nall, 3], dtype=np.float32)
        elif type == "groups":
            s["GroupLen"] = np.zeros([Nall, 1], dtype=np.int32)
            s["GroupMass"] = np.zeros([Nall, 1], dtype=np.float32)
            s["GroupPos"] = np.zeros([Nall,3], dtype=np.float32)
            s["GroupVel"] = np.zeros([Nall,3], dtype=np.float32)
            s["GroupID"] = np.zeros([Nall, 1], dtype=np.int64)
        else:
            print("Type not supported")
        return s

    def __ReadBlock__(self, file:io.BufferedReader, length:int=0):
        head = file.read(4)
        if len(head) == 0:
            return None
        nBytes = length if length > 0 else struct.unpack('i', head)[0]
        res = file.read(nBytes)
        
        check = file.read(4)
        # if length <= 0:
        nBytesCheck = struct.unpack('i', check)[0]
        if (nBytes != nBytesCheck):
            print("head has Bytes", nBytes, ", but tail Bytes", nBytesCheck)
            if length <= 0:
                raise ValueError("invalid check length")
        if self.deBugFlag:
            print("------Read Block with length", nBytes, "------")
        return res, nBytesCheck
    
    def __WriteBlock__(self, file:io.BufferedReader, input:Union[bytes, str, np.ndarray, dict]=None, length:int=0, order='C'):
        """
        Write a Fortran style block of data to file
        Input can be bytes, str, np.ndarray, dict
        Note: dict should not contain str
        """
        binary, nBytes = self.__Input2Bytes(input, length)
        head = struct.pack('i', nBytes)
        file.write(head)
        file.write(binary)
        file.write(head)
        if self.deBugFlag:
            print("------Write Block with length", nBytes, "------")
        return
    
    def __Input2Bytes(self, input:Union[bytes, str, np.ndarray, dict]=None, length:int=0):
        """
        Convert input to bytes
        Note: dict should not contain str
        """
        # loop over all types
        if input is not None:
            if isinstance(input, bytes):
                print("--------------Writing bytes--------------") if self.deBugFlag else None
                binary = input
                nBytes = len(binary)
            elif isinstance(input, str):
                print("--------------Writing str--------------") if self.deBugFlag else None
                binary = input.encode()
                nBytes = length if length > 0 else len(binary)
                binary = struct.pack(f"{nBytes}s", binary)
            elif isinstance(input, np.ndarray):
                print("--------------Writing np.ndarray--------------") if self.deBugFlag else None
                binary = input.tobytes()
                nBytes = len(binary)
            elif isinstance(input, dict):
                binary = b''
                nBytes = 0
                for key in input.keys():
                    print("--------------Writing dict[%s] --------------"%key) if self.deBugFlag else None
                    item = input[key]
                    tmp = self.__Input2Bytes(item)
                    binary += tmp[0]
                    nBytes += tmp[1]
            else:
                print("Type not supported")
        print("input has nBytes:", nBytes) if self.deBugFlag else None
        if length > 0 and nBytes != length:
            print(f"length {length} is not equal to nBytes {nBytes}")
            raise ValueError("invalid length")
        return binary, nBytes

    def __tupleUnpack(self, tuple:tuple, size:int, dict:dict, key=''):
        dict[key] = tuple[0:size]
        tuple = tuple[size:]
        return dict, tuple

    def __BufferUnpack__(self, buffer, count, dtype, size, offset):
        res = np.frombuffer(buffer, dtype, count, offset)
        if count == 1:
            res = res[0]
        offset += size
        return res, offset

    def __SnapBinaryUnpack__(self, s, file, FlagPtype, FlagMass, target, start, length, scale=1):
        start = int(start)
        Npart = s["header"]["Npart"]
        for i, flag in enumerate(FlagPtype):
            if flag:
                Nrows = Npart[i]
                binary, _ = self.__ReadBlock__(file)
                if binary == None:
                    return s, 1 # the end of the file
                if target == "id":
                    # Check length
                    if len(binary) != Nrows * 1 * self.idLen:
                        print("binary length is:", len(binary), "expected length is:", Nrows * 1 * self.idLen)
                        raise ValueError("Binary length doesn't match number of particles")
                    # tmp = np.array(struct.unpack('%sQ'%(Nrows * 1), binary), dtype=np.uint64).reshape(-1, 1)
                    if self.idLen == 8:
                        tmp = np.frombuffer(binary, dtype=np.uint64).reshape(-1, 1)
                    else:
                        tmp = np.frombuffer(binary, dtype=np.uint32).reshape(-1, 1)
                elif target == "":
                    return s, 0 # Pass this block
                else:
                    if len(binary) != Nrows * 3 * 4:
                        # Check length
                        print("binary length is:", len(binary), "expected length is:", Nrows *3 * 4)
                        raise ValueError("Binary length doesn't match number of particles")
                    # tmp = np.array(struct.unpack('%sf'%(Nrows * 3), binary), dtype=np.float32).reshape(-1, 3)
                    tmp = np.frombuffer(binary, dtype=np.float32).reshape(-1, 3) * scale
                s[target][start:start + length[i], :] = tmp
        return s, 0

    def __GroupBinaryUnpack__(self, s, file, target, start, length, scale=1):
        start = int(start)
        Ngroups = s["header"]["Ngroups"]
        binary, _ = self.__ReadBlock__(file)
        
        if binary == None:
            return s, 1 # the end of the file
        
        if target == "GroupLen":
            if len(binary) != Ngroups * 1 * 4:
                print("binary length is:", len(binary), "expected length is:", Ngroups * 1 * 4)
                raise ValueError("Binary length doesn't match number of groups")
            # tmp = np.array(struct.unpack("%si"%(Ngroups), binary), dtype=np.int32).reshape(-1, 1)
            tmp = np.frombuffer(binary, dtype=np.int32).reshape(-1, 1)
        elif target == "GroupMass":
            if len(binary) != Ngroups * 1 * 4:
                print("binary length is:", len(binary), "expected length is:", Ngroups * 1 * 4)
                raise ValueError("Binary length doesn't match number of groups")
            # tmp = np.array(struct.unpack("%sf"%(Ngroups), binary), dtype=np.float32).reshape(-1, 1)
            tmp = np.frombuffer(binary, dtype=np.float32).reshape(-1, 1)
        elif target == "GroupID":
            if len(binary) != Ngroups * 1 * 8:
                print("binary length is:", len(binary), "expected length is:", Ngroups * 1 * 8)
                raise ValueError("Binary length doesn't match number of groups")
            # tmp = np.array(struct.unpack("%sQ"%(Ngroups), binary), dtype=np.int64).reshape(-1, 1)
            tmp = np.frombuffer(binary, dtype=np.int64).reshape(-1, 1)
        elif target == "":
            return s, 0 # Pass this block
        else: # target == "GroupPos" or "GroupVel"
            if len(binary) != Ngroups * 3 * 4:
                print("binary length is:", len(binary), "expected length is:", Ngroups * 3 * 4)
                raise ValueError("Binary length doesn't match number of groups")
            # tmp = np.array(struct.unpack("%sf"%(Ngroups * 3), binary), dtype=np.float32).reshape(-1, 3)
            tmp = np.frombuffer(binary, dtype=np.float32).reshape(-1, 3) * scale
        s[target][start:start + length, :] = tmp
        return s, 0
    
    def __ReadHeader(self, file:io.BufferedReader, type:str):
        """numpy.frombuffer is much faster than struct.unpack"""
        dict = {}
        binary, _ = self.__ReadBlock__(file)
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        # Read Head
        if type == "snapshot":
            header = struct.unpack('6I6d2d2i6i2i4d2i6ii60s', binary) # 96s are unused 96 bytes
            dict, header = self.__tupleUnpack(header, 6, dict, "Npart")
            dict, header = self.__tupleUnpack(header, 6, dict, "Massarr")
            dict, header = self.__tupleUnpack(header, 1, dict, "Time")
            dict, header = self.__tupleUnpack(header, 1, dict, "Redshift")
            dict, header = self.__tupleUnpack(header, 1, dict, "FlagSfr")
            dict, header = self.__tupleUnpack(header, 1, dict, "FlagFeedback")
            dict, header = self.__tupleUnpack(header, 6, dict, "Nall")
            dict, header = self.__tupleUnpack(header, 1, dict, "FlagCooling")
            dict, header = self.__tupleUnpack(header, 1, dict, "NumFiles")
            dict, header = self.__tupleUnpack(header, 1, dict, "BoxSize")
            dict, header = self.__tupleUnpack(header, 1, dict, "Omega0")
            dict, header = self.__tupleUnpack(header, 1, dict, "OmegaLambda")
            dict, header = self.__tupleUnpack(header, 1, dict, "HubbleParam")
            dict, header = self.__tupleUnpack(header, 1, dict, "FlagAge")
            dict, header = self.__tupleUnpack(header, 1, dict, "FlagMetals")
            dict, header = self.__tupleUnpack(header, 6, dict, "NallHW")
            dict = self.__tupleUnpack(header, 1, dict, "flag_entr_ics")[0]
            # ----------------------------------------------------#
            # If last 60 Bytes are used, add more elements here 
            # ----------------------------------------------------#
        elif type == "groups":
            header = struct.unpack('6qi3d', binary) # 80 Bytes in total
            dict, header = self.__tupleUnpack(header, 1, dict, "Ngroups")
            dict, header = self.__tupleUnpack(header, 1, dict, "Nsubhalos")
            dict, header = self.__tupleUnpack(header, 1, dict, "Nids")
            dict, header = self.__tupleUnpack(header, 1, dict, "NgroupsTot")
            dict, header = self.__tupleUnpack(header, 1, dict, "NsubhalosTot")
            dict, header = self.__tupleUnpack(header, 1, dict, "NidsTot")
            dict, header = self.__tupleUnpack(header, 1, dict, "NumFiles")
            dict, header = self.__tupleUnpack(header, 1, dict, "Time / a")
            dict, header = self.__tupleUnpack(header, 1, dict, "Redshift")
            dict = self.__tupleUnpack(header, 1, dict, "BoxSize")[0]
        return dict
    
def ReadBuffer(file:io.BufferedReader, size:int, dtype):
    """
    Raed buffer from file with size and dtype
    """
    buffer = file.read(size)
    res = np.frombuffer(buffer, dtype)
    return res

def ReadDTFE(filename, weight=True, order='C', debug=False):
    s = dict()
    with open(filename, "rb") as f:
        Npart = ReadBuffer(f, 4, np.int32)[0]
        s["BoxSize"] = ReadBuffer(f, 24, np.float32)
        print(s["BoxSize"]) if debug else None
        s["pos"] = ReadBuffer(f, Npart*3*4, np.float32).reshape((-1, 3), order=order)
        print(s["pos"]) if debug else None
        if weight:
            s["weight"] = ReadBuffer(f, Npart*4, np.float32).reshape((-1, 1), order=order)
            print(s["weight"]) if debug else None
        s["vel"] = ReadBuffer(f, Npart*3*4, np.float32).reshape((-1, 3), order=order)
        print(s["vel"]) if debug else None
    # Check length
    for key in s.keys():
        if key == "BoxSize":
            pass
        elif len(s[key]) != Npart:
            print("Warning: %s length does not match"%(key))
    return s

def WriteDTFE(filename, Npart, BoxSize, pos, vel=None, weight=None, DustType=None, DustPos=None, Ndust=0, WeightDust=1., order='C'):
    """
    Write catalogue to DTFE type, with data type np.float32
    DustType: None, "Uniform", "Generated"
    WeightDust is set to 1. as default, halo weight should adjust according to grplength, support numpy array
    weight and WeightDust are np.float32
    """
    with open(filename, "wb") as f:
        # Check data
        if (Npart != len(pos)): 
            raise ValueError(f"Pos should have length Nall={Npart} instead of len(pos)={len(pos)}")
        print("------------------------------------------")
        print(f"Writing to {filename}")
        f.write(np.array([Npart + Ndust], dtype=np.int32).tobytes(order))
        f.write(np.array([0., BoxSize, 0., BoxSize, 0., BoxSize], dtype=np.float32).tobytes(order))
        # Write pos
        if DustType == "Uniform":
            tmp = np.random.random([Ndust, 3]).astype(np.float32) * BoxSize
            pos = np.vstack((pos, tmp))
        elif DustType == "Generated":
            pos = np.vstack((pos, DustPos)).astype(np.float32)
        elif DustType is not None:
            raise ValueError(f"Unknown DustType {DustType}")
        # print(pos)
        f.write(pos.tobytes(order))
        # Write weight
        if weight is not None:
            if (Npart != len(weight)):
                raise ValueError(f"Weight should have length Nall={Npart} instead of len(weight)={len(weight)}")
            weight = weight.astype(np.float32)
            # if DustType is not None:
            #     try:
            #         if (Ndust != len(WeightDust)):
            #             raise ValueError(f"WeightDust should have length Ndust={Ndust} instead of len(WeightDust)={len(WeightDust)}")
            #     except:
            #         WeightDust = WeightDust * np.ones([Ndust, 1], dtype=np.int32)
                # weight = np.vstack((weight, WeightDust))
            # print(weight)
            # f.write(weight.tobytes(order))
        else:
            weight = np.ones(Npart, dtype=np.float32)
            print("Warning: particle weight is set to default value 1.")
        if DustType is not None:
            try:
                if (Ndust != len(WeightDust)):
                    raise ValueError(f"WeightDust should have length Ndust={Ndust} instead of len(WeightDust)={len(WeightDust)}")
            except:
                WeightDust = WeightDust * np.ones([Ndust, 1], dtype=np.float32)
            weight = np.vstack((weight, WeightDust))
        f.write(weight.tobytes(order))
        # Write vel
        if vel is not None:
            f.write(vel.tobytes(order))
            if DustType is not None:
                f.write(np.zeros([Ndust, 3], dtype=np.float32))
        else:
            vel = np.zeros([Npart + Ndust, 3], dtype=np.float32)
        
        print(f"Finished!")
        print("------------------------------------------")
        
def WriteSamplePoints(filename, Nsamples, pos, cellsize=None):
    with open(filename, "wb") as f:
        f.write(np.array([Nsamples], dtype=np.int32).tobytes())
        f.write(pos.tobytes())
        if cellsize is None:
            cellsize = np.zeros([Nsamples, 3], dtype=np.float32)
        f.write(cellsize.tobytes())
        
def getParticleDisp(ppos1, ppos2, pid1, pid2, boxSize):
    sort1 = np.argsort(pid1, axis=0)[:, 0]
    tmp = np.take(sort1, pid2.astype("int")[:,0]-1, axis=0)
    tmp = np.take(ppos1, tmp, axis=0)
    disp = ppos2 - tmp
    
    disp[disp > boxSize / 2] = disp[disp > boxSize / 2] - boxSize
    disp[disp < -boxSize / 2] = disp[disp < -boxSize / 2] + boxSize
    return disp, tmp

def getHaloParticleDisp(ppos1, pid1, pid2, grppos2, grplen2, boxSize):
    sort1 = np.argsort(pid1, axis=0)[:, 0]
    
    hpdisp = np.zeros([np.sum(grplen2), 3], dtype=np.float32)
    start = 0
    for i, grplen in enumerate(grplen2[:, 0]):
        tmp_id = pid2[start:start + grplen]
        tmp_pos = ppos1[sort1[tmp_id[:, 0] - 1]]
        hpdisp[start:start + grplen, 0:3] = grppos2[i] - tmp_pos
        start += grplen
    hpdisp[hpdisp > boxSize / 2] = hpdisp[hpdisp > boxSize / 2] - boxSize
    hpdisp[hpdisp < -boxSize / 2] = hpdisp[hpdisp < -boxSize / 2] + boxSize
    # Displacement is ordered as pid2
    return hpdisp, ppos1[sort1[pid2[0:np.sum(grplen2)] - 1]]

def getHaloDisp(ppos1, pid1, pid2, grppos2, grplen2, boxSize, grpmass2=None):
    sort1 = np.argsort(pid1, axis=0)[:, 0]
    # sort2 = np.argsort(pid2, axis=0)[:, 0]
    grppos1 = np.zeros([len(grplen2), 3], dtype=np.float32)
    grpdisp = np.zeros([len(grplen2), 3], dtype=np.float32)
    start = 0
    for i, grplen in enumerate(grplen2[:, 0]):
        tmp_id = pid2[start:start + grplen]
        tmp_pos = ppos1[sort1[tmp_id[:, 0] - 1]]
        tmp_disp = grppos2[i] - tmp_pos
        
        tmp_disp[tmp_disp > boxSize / 2] = tmp_disp[tmp_disp > boxSize / 2] - boxSize
        tmp_disp[tmp_disp < -boxSize / 2] = tmp_disp[tmp_disp < -boxSize / 2] + boxSize
        
        grpdisp[i] = np.sum(tmp_disp, axis=0) / grplen
        grppos1[i] = (grppos2[i] - grpdisp[i]) % boxSize
        start += grplen
    if grpmass2 is not None:
        return grpdisp, grppos1, grpmass2
    return grpdisp, grppos1

def getInHaloDisp(ppos1, ppos2, pid1, pid2, grplen2, boxSize):
    sort1 = np.argsort(pid1, axis=0)[:, 0]
    tmp_pid2 = pid2[0:np.sum(grplen2)]
    tmp_pid1 = pid1[sort1[tmp_pid2[:, 0] - 1]]
    tmp_ppos2 = ppos2[0:np.sum(grplen2)]
    tmp_ppos1 = ppos1[sort1[tmp_pid2[:, 0] - 1]]
    return getParticleDisp(tmp_ppos1, tmp_ppos2, tmp_pid1, tmp_pid2, boxSize)

def getNotHaloDisp(ppos1, ppos2, pid1, pid2, grplen2, boxSize):
    sort1 = np.argsort(pid1, axis=0)[:, 0]
    tmp_pid2 = pid2[np.sum(grplen2):]
    tmp_pid1 = pid1[sort1[tmp_pid2[:, 0] - 1]]
    tmp_ppos2 = ppos2[np.sum(grplen2):]
    tmp_ppos1 = ppos1[sort1[tmp_pid2[:, 0] - 1]]
    return getParticleDisp(tmp_ppos1, tmp_ppos2, tmp_pid1, tmp_pid2, boxSize)

# Here defines some structs used in above functions

class NDskl_node_str:
    def __init__(self):
        self.pos_index:np.int32 = -1
        self.flags:np.int32 = -1
        self.nnext:np.int32 = -1
        self.type:np.int32 = -1
        self.index:np.int32 = -1
        self.nsegs = np.array([], dtype=np.int32)
        self.nextNode = np.array([], dtype=np.int32)
        self.nextSeg = np.array([], dtype=np.int32)
        
    def __dict__(self):
        return {"pos_index":self.pos_index, \
                "flags":self.flags, \
                "nnext":self.nnext, \
                "type":self.type, \
                "index":self.index, \
                "nsegs":self.nsegs, \
                "nextNode":self.nextNode, \
                "nextSeg":self.nextSeg}
    
    def help(self):
        print("The NDskl_node_str contains the following elements:")
        print("pos_index: the index of the node in the nodepos/nodedata array")
        print("flags: the flags of the node (identify boundary nodes)")
        print("nnext: number of connected arcs")
        print("type: critical index (ndims+1 for bifurcations / trimmed arc axtremity)")
        print("index: index of this node")
        print("nsegs: number of segments in each connected arc")
        print("nextNode: index of the other node of the arc")
        print("nextSeg: index of the first segment on the arc")
        return
    
    def __checkValid__(self):
        if self.pos_index < 0:
            raise ValueError("pos_index is not valid")
        if self.flags < 0:
            raise ValueError("flags is not valid")
        if self.nnext < 0:
            raise ValueError("nnext is not valid")
        if self.type < 0:
            raise ValueError("type is not valid")
        if self.index < 0:
            raise ValueError("index is not valid")
        if self.nsegs.size != self.nnext:
            raise ValueError("nsegs length does not match nnext")
        if self.nextNode.size != self.nnext:
            raise ValueError("nextNode length does not match nnext")
        if self.nextSeg.size != self.nnext:
            raise ValueError("nextSeg length does not match nnext")

class NDskl_seg_str:
    def __init__(self):
        self.pos_index:np.int32 = -1
        self.nodes = [-1, -1]
        self.flags:np.int32 = -1
        self.index:np.int32 = -1
        self.next_seg:np.int32 = -1
        self.prev_seg:np.int32 = -1
    
    def __dict__(self):
        return {"pos_index":self.pos_index, \
                "nodes":self.nodes, \
                "flags":self.flags, \
                "index":self.index, \
                "next_seg":self.next_seg, \
                "prev_seg":self.prev_seg}
        
    def help(self):
        print("The NDskl_seg_str contains the following elements:")
        print("pos_index: index in the segpos/segdata arrays")
        print("nodes: index of the 2 nodes at the endpoints of the current arc")
        print("flags: flags (identify boundary nodes)")
        print("index: index of this segment")
        print("next_seg: index of the next segment in the node (-1 if none)")
        print("prev_seg: index of the previous segment in the node (-1 if none)")
        return
    
    def __checkValid__(self):
        if self.pos_index < 0:
            raise ValueError("pos_index is not valid")
        if self.nodes[0] < 0 or self.nodes[1] < 0:
            raise ValueError("nodes is not valid")
        if self.flags < 0:
            raise ValueError("flags is not valid")
        if self.index < 0:
            raise ValueError("index is not valid")
        if self.next_seg < -1:
            raise ValueError("next_seg is not valid")
        if self.prev_seg < -1:
            raise ValueError("prev_seg is not valid")