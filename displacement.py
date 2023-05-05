from typing_extensions import Self
import numpy as np
import sys
import os
import struct
import io

################################
"""
对于大量二进制文件的读取，如果需要将结果转化成numpy.ndarray，最好的方式是直接使用numpy.frombuffer(推荐)或者numpy.ndarray(buffer)
如果先通过struct,unpack读取tuple再转化成为numpy.ndarray的时间、空间消耗会远远大于上述方法
时间消耗：numpy.frombuffer < numpy.ndarray(buffer) < struct.unpack < np.array(struct.unpack())
参考链接：https://stackoverflow.com/questions/54679949/unpacking-binary-file-using-struct-unpack-vs-np-frombuffer-vs-np-ndarray-vs-np-f/54681417#54681417?newreg=320f8a5066014cbd925bdc007897b523
""" 
################################
class ReadGadget():
    def __init__(self, datadir):
        self.simudir = datadir

    def setPath(self, dirname, filename):
        self.name = filename
        self.dirpath = self.simudir + "/" + dirname
        self.filename = self.dirpath + "/" + filename

    def ReadGadget(self, type, target, FlagPtype=np.array([False, True, False, False, False, False]), deBugFlag=True, nFiles=-1, MpcUnit=1000):
        totalLen = len(os.listdir(self.dirpath))
        if deBugFlag == True:
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
                if deBugFlag:
                    print("--------------Opening %s.%s --------------"%(self.name, fid))
                    print("--------------Reading Header--------------")
                s["header"] = self.ReadHeader(f, type)
                Nall = np.sum(s["header"]['Nall'][FlagPtype]) if type=="snapshot" else s["header"]['NgroupsTot']
                if fid == 0:
                    s = self.__genDict__(s, Nall, type)
                if type == "snapshot":
                    length = s["header"]["Npart"] * FlagPtype
                    Massarr = s["header"]['Massarr']
                    FlagMass = (Massarr > 0) & FlagPtype
                    for item in ["pos", "vel", "id"]:
                        s = self.__ReadBody__(s, f, type, item, FlagPtype, FlagMass, start, length, deBugFlag, scale)
                    start = start + np.sum(length)
                elif type == "groups":
                    length = s["header"]["Ngroups"]
                    for item in ["GroupLen", "GroupMass", "GroupPos", "", "GroupID"]:
                        s = self.__ReadBody__(s, f, type, item, None, None, start, length, deBugFlag, scale)
                    start = start + length
                if deBugFlag:
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

    def ReadHeader(self, file:io.BufferedReader, type:str):
        dict = {}
        binary = self.__ReadBlock__(file)
        if binary == None: # Check Validity
            raise ValueError("no bytes can be read")
        offset = 0
        if type == "snapshot":
            # header = struct.unpack('6I6d2d2i6i2i4d2i6ii60s', binary) # 96s are unused 96 bytes
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
    
    def __ReadBody__(self, s:dict, file:io.BufferedReader, type:str, target:str, FlagPtype, FlagMass, start, length, deBugFlag, scale):
        if deBugFlag:
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

    def __ReadBlock__(self, file:io.BufferedReader):
        head = file.read(4)
        if len(head) == 0:
            return None
        nBytes = struct.unpack('i', head)[0]
        res = file.read(nBytes)
        
        check = file.read(4)
        nBytesCheck = struct.unpack('i', check)[0]
        if (nBytes != nBytesCheck):
            print("head Bytes", nBytes, "tail Bytes", nBytesCheck)
            raise ValueError("invalid check length")
        return res

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
                binary = self.__ReadBlock__(file)
                if binary == None:
                    return s, 1 # the end of the file
                if target == "id":
                    # Check length
                    if len(binary) != Nrows * 1 * 8:
                        print("binary length is:", len(binary), "expected length is:", Nrows * 1 * 8)
                        raise ValueError("Binary length doesn't match number of particles")
                    # tmp = np.array(struct.unpack('%sQ'%(Nrows * 1), binary), dtype=np.uint64).reshape(-1, 1)
                    tmp = np.frombuffer(binary, dtype=np.uint64).reshape(-1, 1)
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
        binary = self.__ReadBlock__(file)
        
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
        binary = self.__ReadBlock__(file)
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
    buffer = file.read(size)
    res = np.frombuffer(buffer, dtype)
    return res

def ReadDTFE(filename, weight=True):
    s = dict()
    with open(filename, "rb") as f:
        Npart = ReadBuffer(f, 4, np.int32)[0]
        s["BoxSize"] = ReadBuffer(f, 24, np.float32)
        s["pos"] = ReadBuffer(f, Npart*3*4, np.float32).reshape(-1, 3)
        if weight:
            s["weight"] = ReadBuffer(f, Npart*4, np.float32).reshape(-1, 1)
        s["vel"] = ReadBuffer(f, Npart*3*4, np.float32).reshape(-1, 3)
    # Check length
    for key in s.keys():
        if key == "BoxSize":
            pass
        elif len(s[key]) != Npart:
            print("Warning: %s length does not match"%(key))
    return s

def WriteDTFE(filename, Nall, BoxSize, pos, vel, weight=None):
    with open(filename, "wb") as f:
        f.write(np.array([Nall], dtype=np.int32).tobytes())
        f.write(np.array([0., BoxSize, 0., BoxSize, 0., BoxSize], dtype=np.float32).tobytes())
        f.write(pos.tobytes())
        if weight is not None:
            f.write(weight.tobytes())
        else:
            weight = np.ones(Nall, dtype=np.float32)
            f.write(weight.tobytes())
        f.write(vel.tobytes())
        
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