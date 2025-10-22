import os
import struct
import torch
import numpy as np
import bitarray as ba
from utils.option import dict2nonedict

"""
% The MRC image header has a fixed size of 1024 bytes. 
The information within the header includes a description of the extended header and image data. 
The column, row, and section are equivalent to the x, y, and z axes, namely [Width, Height, Depth] in terms of Image Process.
% 
% byte    
% 以下为大端顺序
% Numbers  Variable Type Variable Name	  Contents
%  1 - 4	      i	     NumCol	            Number of columns. Typically, the number of image elements along the X axis.
%  5 - 8	      i	     NumRow	            Number of rows. Typically, the number of image elements along the Y axis.
% 9 - 12	      i	     NumSections	    Total number of sections. (NumZSec*NumWave*NumTimes)
% 13 - 16	      i    	PixelType	        The format of each pixel value. See the Pixel Data Types table below.
% 17 - 20	      i	     mxst	            Starting point along the X axis of sub-image (pixel number). Default is 0.
% 21 - 24	      i	     myst	            Starting point along the Y axis of sub-image (pixel number). Default is 0.
% 25 - 28	      i   	 mzst	            Starting point along the Z axis of sub-image (pixel number). Default is 0.
% 29 - 32	      i	     mx	                Sampling frequency in x; commonly set equal to one or the number of columns.
% 33 - 36	      i	     my	                Sampling frequency in y; commonly set equal to one or the number of rows.
% 37 - 40	      i	     mz	                Sampling frequency in z; commonly set equal to one or the number of z sections.
% 41 - 44	      f	     dx	                Cell dimension in x; for non-crystallographic data, 
                                            set to the x sampling frequency times the x pixel spacing.
% 45 - 48	      f	     dy	                Cell dimension in y; for non-crystallographic data, 
                                            set to the y sampling frequency times the y pixel spacing.
% 49 - 52	      f    	 dz	                Cell dimension in z; for non-crystallographic data, 
                                            set to the z sampling frequency times the z pixel spacing.
% 53 - 56	      f	     alpha	            Cell angle (alpha) in degrees. Default is 90.
% 57 - 60	      f	     beta	            Cell angle (beta) in degrees. Default is 90.
% 61 - 64	      f	     gamma	            Cell angle (gamma) in degrees. Default is 90.
% 65 - 68	      i	     -	                Column axis. Valid values are 1,2, or 3. Default is 1.
% 69 - 72	      i	     -	                Row axis. Valid values are 1,2, or 3. Default is 2.
% 73 - 76	      i	     -	                Section axis. Valid values are 1,2, or 3. Default is 3.
% 77 - 80	      f	     min	            Minimum intensity of the 1st wavelength image.
% 81 - 84	      f	     max	            Maximum intensity of the 1st wavelength image.
% 85 - 88	      f	     mean	            Mean intensity of the first wavelength image.
% 89 - 92	      i	     nspg	            Space group number. Applies to crystallography data.
% 93 - 96	      i	     next	            Extended header size, in bytes.
% 97 - 98	      n	     dvid	            ID value. (-16224)
% 99 - 100	      n	     nz	                Unused. (1)
% 101 - 104	      i	     ntst	            Starting time index.
% 105 - 128	      c24	 blank	            Blank section. 24 bytes.
% 129 - 130       n	     NumIntegers	    Number of 4 byte integers stored in the extended header per section.
% 131 - 132	      n	     NumFloats	        Number of 4 byte floating-point numbers stored in the extended header per section.
% 133 - 134       n	     sub	            Number of sub-resolution data sets stored within the image. Typically, this equals 1.
% 135 - 136	      n	     zfac	            Reduction quotient for the z axis of the sub-resolution images.
% 137 - 140	      f	     min2	            Minimum intensity of the 2nd wavelength image.
% 141 - 144	      f	     max2	            Maximum intensity of the 2nd wavelength image.
% 145 - 148	      f	     min3	            Minimum intensity of the 3rd wavelength image.
% 149 - 152	      f	     max3	            Maximum intensity of the 3rd wavelength image.
% 153 - 156	      f	     min4	            Minimum intensity of the 4th wavelength image.
% 157 - 160       f	     max4	            Maximum intensity of the 4th wavelength image.
% 161 - 162	      n	     type	            Image type. See the Image Type table below.(not used now)
% 163 - 164	      n	     LensNum	        Lens identification number. 
                                            Olympus: 1=1.7 NA, 2=1.5NA, 3=1.35NA,  Nikon: 4=1.49NA, 5=1.35NA
% 165 - 166	      n	     n1	                Depends on the image type.
% 167 - 168	      n	     n2	                Depends on the image type.
% 169 - 170	      n	     v1	                Depends on the image type.
% 171 - 172	      n	     v2	                Depends on the image type.
% 173 - 176	      f	     min5	            Minimum intensity of the 5th wavelength image.
% 177 - 180	      f	     max5	            Maximum intensity of the 5th wavelength image.
% 181 - 182	      n	     NumTimes	        Number of time points.
% 183 - 184	      n	     ImgSequence	    Image sequence. 0=ZTW, 1=WZT, 2=ZWT.
% 185 - 188	      f	     -	                X axis tilt angle (degrees).
% 189 - 192	      f	     -	                Y axis tilt angle (degrees).
% 193 - 196	      f	     -	                Z axis tilt angle (degrees).
% 197 - 198	      n	     NumWaves	        Number of em wavelengths.
% 199 - 200	      n	     wave1	            Wavelength 1, in nm. must be in [525, 610, 660]
% 201 - 202	      n	     wave2	            Wavelength 2, in nm. must be zero
% 203 - 204	      n	     wave3	            Wavelength 3, in nm. must be zero
% 205 - 206	      n	     wave4	            Wavelength 4, in nm. must be zero
% 207 - 28	      n	     wave5	            Wavelength 5, in nm. must be zero
% 209 - 212	      f	     z0	                Z origin, in um.
% 213 - 216	      f	     x0	                X origin, in um.
% 217 - 220	      f	     y0	                Y origin, in um.
% 221 - 224	      i	     NumTitles	        Number of titles. Valid numbers are between 0 and 10.
% 225 - 226       n      SystemType         System type: 0=LS(Light Sheet), 1=SIM, 2=ISM, 3=WF(Wide Field).
% 227 - 228       n      ImagingMode        
% 229 - 304	      c80	 -	                Title 1. 80 characters long.
% 305 - 384	      c80	 -	                Title 2. 80 characters long.
% 385 - 464	      c80	 -	                Title 3. 80 characters long.
% 465 - 544	      c80	 -	                Title 4. 80 characters long.
% 545 -624	      c80	 -	                Title 5. 80 characters long.
% 625-704	      c80	 -	                Title 6. 80 characters long.
% 705-784	      c80	 -	                Title 7. 80 characters long.
% 785-864	      c80	 -	                Title 8. 80 characters long.
% 865-944	      c80	 -         	        Title 9. 80 characters long.
% 945-1024	      c80	 -	                Title 10. 80 characters long.

% Pixel Data Types
% The data type of an image, stored in header bytes 13-16, is designated by one of the code numbers in the following table. 
%
% Code	C/C++ Macro	        Description
%  0	IW_BYTE	            1-byte unsigned integer
%  1	IW_SHORT	        2-byte signed integer
%  2	IW_FLOAT	        4-byte floating-point (IEEE)
%  3	IW_COMPLEX_SHORT	4-byte complex value as 2 2-byte signed integers
%  4	IW_COMPLEX	        8-byte complex value as 2 4-byte floating-point values
%  5	IW_EMTOM	        2-byte signed integer
%  6	IW_USHORT	        2-byte unsigned integer
%  7	IW_LONG	            4-byte signed integer

"""


# TODO
"""
TODO: 实现MMAP加速IO
https://www.cnblogs.com/zhoujinyi/p/6062907.html
https://bbs.huaweicloud.com/blogs/313265

去掉correct_header函数
"""

HEADER_SIZE = 1024  # 1024 bytes
RESULTANT_NORM_INTENSITY = 1000.0  # 每帧的前1%像素均值被正规化到1000(uint16)或1.0(float32)


# ----------------------------------------
# Function: Correct header | for otf | 实际上是把大端改为了小端的读取方式
# ----------------------------------------
def correct_header(header, big_endian=False):
    if isinstance(header, tuple): header = list(header)
    if header[24] % 65536 == 49312:
        if big_endian is False:
            pass
        else:
            print("header is wrong and we change the header, check the code!")
            for idx in [24, 41, 45, 49, 50]:
                header[idx] = header[idx] % 65536 * 65536 + header[idx] // 65536
    elif header[24] // 65536 == 49312:
        if big_endian is True:
            pass
        else:
            print("header is wrong and we change the header, check the code!")
            for idx in [24, 41, 45, 49, 50]:
                header[idx] = header[idx] % 65536 * 65536 + header[idx] // 65536
    else:
        print("header is wrong and can not be read, check the mrc file header!")
        raise IOError
    return header


# ----------------------------------------
# Function: make header
# 从小端的RAW Header中推出小端的RAW/WF/SIM Header
# ----------------------------------------
def make_sr_header(header, opt):

    if isinstance(header, tuple): header = list(header)
    header[41] = 65536 * 1 + 1  # orientation = 1, phase = 1
    header[45] = 65536 * 0 + 0  # data_save_order = 0, num_timepoint = 0
    header[2] = 0 # num_timepoint = 0  ->  num_section = 0
    if opt['num_step'] == 2:
        header[57] = 2 * 65536 + header[57] % 65536  # step=2
    else:
        header[57] = 65536 + header[57] % 65536  # step=1

    header[0] = opt['num_pixel_width'] * opt['zoom_factor_xy']
    header[1] = opt['num_pixel_height'] * opt['zoom_factor_xy']
    header[24] = opt['num_pixel_depth'] * opt['zoom_factor_z'] * 65536 + 49312

    # header[2] = opt['num_pixel_depth'] * opt['num_timepoint'] * opt['num_channel'] * 1 * 1  # num_sections = num_pixel_depth * num_timepoint * num_channel * orientation * phase

    header[10] = struct.unpack('I', struct.pack('f', opt['width_space_sampling'] / opt['zoom_factor_xy']))[0]
    header[11] = struct.unpack('I', struct.pack('f', opt['height_space_sampling'] / opt['zoom_factor_xy']))[0]
    header[12] = struct.unpack('I', struct.pack('f', opt['depth_space_sampling'] / opt['zoom_factor_z']))[0]
    header[3] = 2  # single as default

    return header

def make_wf_header(header, opt):

    if isinstance(header, tuple): header = list(header)
    header[41] = 65536 * 1 + 1  # orientation = 1, phase = 1
    header[45] = 65536 * 0 + 0  # data_save_order = 0, num_timepoint = 0
    header[2] = 0 # num_timepoint = 0  ->  num_section = 0
    if opt['num_step'] == 3:
        header[57] = 3 * 65536 + 3  # step=3
    elif opt['num_step'] == 2:
        # header[57] = 2 * 65536 + header[57] % 65536  # step=2
        header[57] = 2 * 65536 + 2
    else:
        # header[57] = 65536 + header[57] % 65536  # step=1
        header[57] = 65536 + 1

    header[24] = opt['num_pixel_depth'] * 65536 + 49312

    header[3] = 2  # single as default

    return header

def make_sr_header_assign_scale(header, opt, scale):  # 适用于opt不带scale参数的情况

    if isinstance(header, tuple): header = list(header)
    header[41] = 65536 * 1 + 1  # orientation = 1, phase = 1
    header[45] = 65536 * 0 + 0  # data_save_order = 0, num_timepoint = 0
    header[2] = 0
    if opt['num_step'] == 3:
        header[57] = 3 * 65536 + 3  # step=3
    elif opt['num_step'] == 2:
        # header[57] = 2 * 65536 + header[57] % 65536  # step=2
        header[57] = 2 * 65536 + 2
    else:
        # header[57] = 65536 + header[57] % 65536  # step=1
        header[57] = 65536 + 1

    header[0] = opt['num_pixel_width'] * scale[0]
    header[1] = opt['num_pixel_height'] * scale[1]
    header[24] = opt['num_pixel_depth'] * scale[2] * 65536 + 49312

    # header[2] = opt['num_pixel_depth'] * opt['num_timepoint'] * opt['num_channel'] * 1 * 1  # num_sections = num_pixel_depth * num_timepoint * num_channel * orientation * phase

    header[10] = struct.unpack('I', struct.pack('f', opt['width_space_sampling'] / scale[0]))[0]
    header[11] = struct.unpack('I', struct.pack('f', opt['height_space_sampling'] / scale[1]))[0]
    header[12] = struct.unpack('I', struct.pack('f', opt['depth_space_sampling'] / scale[2]))[0]
    header[3] = 2  # single

    return header


# ----------------------------------------
# Class: ReadMRC
# ----------------------------------------
# Function:
#       -> read header and option
#       -> read sim raw .mrc files
#       -> read general .mrc files including otf
#       -> batch reading (deprecated for the lack of ending signal)
#       -> Format as [timepoint, (num_orientation, num_phase, num_wave), depth, height, width] array
# API:
#       -> __init__ read header and get option
#       -> read some data and convert to [tuple, list, torch]
#       -> read next batch
# [safe]:
#    * do not deliver pointer (handle) *
#    * control pointer scope with 'with' *
# ----------------------------------------

class ReadMRC:
    def __init__(self, file, opt=None, is_SIM_rawdata=True, big_endian=None):

        """
        is_SIM_rawdata: if False, force num_of_orientation-num_of_phase to 1-1
        """

        if os.path.exists(file):
            if os.path.getsize(file) < 1024:
                print("header incomplete")
                raise RuntimeError

        else:
            print("cannot find {}".format(file))
            raise FileNotFoundError

        # make public vars

        self.file = file

        self.is_SIM_rawdata = is_SIM_rawdata

        # process

        self.header, self.big_endian = self.__read_mrc_header(big_endian)

        self.big_endian_signal = '>' if self.big_endian else '<'

        self.opt = self.__get_option_from_mrc_header(opt)

        self.timepoint_have_been_read = 0

        self.totalsize = 0

    # ----------------------------------------
    # read header
    # ----------------------------------------
    def __read_mrc_header(self, big_endian=None):

        file = self.file

        # auto choose big_endian / small endian type
        if big_endian is None:
            big_endian = False
            with open(file, 'rb') as f:
                header = struct.unpack('<256I', f.read(HEADER_SIZE))
            if header[3] > 12:
                big_endian = True
                with open(file, 'rb') as f:
                    header = struct.unpack('>256I', f.read(HEADER_SIZE))
        # < < < < < < < < < < < < <

        # small_endian type > > > >
        elif big_endian is True:
            with open(file, 'rb') as f:
                header = struct.unpack('>256I', f.read(HEADER_SIZE))
        # < < < < < < < < < < < < <

        # big_endian type > > > > >
        elif big_endian is False:
            with open(file, 'rb') as f:
                header = struct.unpack('<256I', f.read(HEADER_SIZE))
        # < < < < < < < < < < < < <

        # data offset (pos of data) should be 1024
        assert header[23] == 0, "data offset should be zero! update the code if necessary"

        if isinstance(header, tuple): header=list(header)

        return header, big_endian

    # ----------------------------------------
    # header -> imaging options
    # ----------------------------------------
    def __get_option_from_mrc_header(self, opt):

        if opt is None: opt = dict2nonedict({})
        header = self.header

        basename = os.path.basename(self.file).lower()

        if '3d-sim' in basename:
            opt['mode'] = '3d'
        elif 'tirf-sim' in basename:
            opt['mode'] = 'tirf'
        elif 'highnagi-sim' in basename or 'highna_gi-sim' in basename or 'high na gi-sim' in basename:
            opt['mode'] = 'highnagi'
        elif 'lownagi-sim' in basename or 'lowna_gi-sim' in basename or 'low na gi-sim' in basename:
            opt['mode'] = 'lownagi'
        else:
            opt['mode'] = None

        opt['num_pixel_width'] = header[0]
        opt['num_pixel_height'] = header[1]

        if '-3step' in basename:
            opt['num_step'] = 3
        elif '-2step' in basename:
            opt['num_step'] = 2
        else:
            temp = header[57] // 65536
            opt['num_step'] = temp if temp in [1, 2] else 1

        # Total number of sections. (NumZSec*NumWave*NumTimes*NumOrient*NumPhase)
        opt['num_section'] = header[2]

        # spacing rate, i.e., pixel size in space domain (in um)
        opt['width_space_sampling'] = struct.unpack('f', struct.pack('I', header[10]))[0]
        opt['height_space_sampling'] = struct.unpack('f', struct.pack('I', header[11]))[0]
        opt['depth_space_sampling'] = abs(struct.unpack('f', struct.pack('I', header[12]))[0])

        opt['is_complex'], opt['byte_per_pixel'], opt['dtype_symbol'] = \
            {'6': [False, 2, 'H'], '4': [True, 8, 'f'], '2': [False, 4, 'f'], '1': [False, 2, 'i'], '0': [False, 1, 'B'], '12': [False, 2, 'e']}[str(header[3])]

        if self.big_endian is False: # labelview data

            # opt['num_channel'] = max(1, struct.unpack('2h', struct.pack('I', header[49]))[0])
            # assert opt['num_channel'] == 1, "multi-color data are not supported yet, check the file header"
            opt['num_channel'] = 1

            if not self.is_SIM_rawdata:
                opt['num_phase'] = max(1, struct.unpack('2h', struct.pack('I', header[49]))[0])
                opt['num_orientation'] = 1
            else:
                # number of phases
                opt['num_phase'] = max(1, header[41] % 65536)
                # number of orientation
                opt['num_orientation'] = max(1, header[41] // 65536)

            em_wavelength = struct.unpack('2h', struct.pack('I', header[49]))[1]
            if em_wavelength == 0: em_wavelength = 450
            opt['em_wavelength'] = [em_wavelength]
            
            # MultiSIM001采集的数据，  timepoint包括step，phase不包括step
            # MultiSIM002采集的数据，  timepoint不包括step，phase包括step
            # MultiSIM003采集的数据，  timepoint不包括step，phase不包括step
            # 为区别，利用脚本将源数据文件头修改为：2step数据的timepoints包括step，3step数据的timepoints不包括step，phase均不包括step
            if opt['num_phase'] == 10:
                opt['num_phase'] = 5
                opt['num_step'] = 2
                # opt['num_timepoint'] = header[45] % 65536 # 超过65536会解析错误
                opt['num_timepoint'] = header[45]
            elif '-3step' in basename or '-2step' in basename:
                # opt['num_timepoint'] = header[45] % 65536 # 超过65536会解析错误
                opt['num_timepoint'] = header[45]
            else:
                # opt['num_timepoint'] = header[45] % 65536 // opt['num_step'] # 超过65536会解析错误
                opt['num_timepoint'] = header[45] // opt['num_step']

            if not self.is_SIM_rawdata and opt['num_timepoint'] <= 0: opt['num_timepoint'] = 1 # OTF

            if '405' in basename:
                opt['ex_wavelength'] = [405]
            elif '445' in basename:
                opt['ex_wavelength'] = [445]
            elif '488' in basename:
                opt['ex_wavelength'] = [488]
            elif '561' in basename:
                opt['ex_wavelength'] = [561]
            elif '640' in basename:
                opt['ex_wavelength'] = [640]
            else:
                # raise RuntimeError
                opt['ex_wavelength'] = [-1]

            try_depth = header[24] // 65536
            if try_depth > 1:
                opt['num_pixel_depth'] = try_depth
                assert_tps = opt['num_section'] // (opt['num_phase'] * opt['num_orientation'] * opt['num_pixel_depth'] * opt['num_channel'] * opt['num_step'])
                if opt['num_timepoint'] == assert_tps:
                    pass
                elif opt['num_timepoint'] == assert_tps + 1:
                    opt['num_timepoint'] -= 1 # labview's bug
                else:
                    print('header may be wrong')
            else:
                if opt['num_timepoint'] == 0:
                    opt['num_pixel_depth'] = 0
                else:
                    temp = opt['num_section'] // (opt['num_phase'] * opt['num_orientation'] * opt['num_timepoint'] * opt['num_channel'] * opt['num_step'])
                    if temp == 0:
                        opt['num_timepoint'] = opt['num_timepoint'] - 1
                        if opt['num_timepoint'] == 0:
                            opt['num_pixel_depth'] = 0
                        else:
                            opt['num_pixel_depth'] = opt['num_section'] // (opt['num_phase'] * opt['num_orientation'] * opt['num_timepoint'] * opt['num_channel'] * opt['num_step'])
                    else:
                        opt['num_pixel_depth'] = temp

            if not self.is_SIM_rawdata and opt['num_pixel_depth'] > 3 and opt['num_pixel_depth'] == opt['num_pixel_width']: opt['num_pixel_depth'] = 3 # OTF

            # data save mode. Z: depth | W: number of wave | T: timepoints
            # opt['data_save_order'] = {'0': 'ZTW', '1': 'WZT', '2': 'ZWT'}[str(header[45] // 65536)]
            opt['data_save_order'] = 'ZTW' # header[45]中另一位超过65536后溢出到这边

        else: # big_endian data

            # opt['num_channel'] = max(1, struct.unpack('2h', struct.pack('I', header[49]))[1])
            # assert opt['num_channel'] == 1, "multi-color data are not supported yet, check the file header"
            opt['num_channel'] = 1

            if not self.is_SIM_rawdata:
                opt['num_phase'] = 1
                opt['num_orientation'] = 1
            else:
                # number of phases
                opt['num_phase'] = max(1, header[41] // 65536)
                # number of orientation
                opt['num_orientation'] = max(1, header[41] % 65536)

            em_wavelength = struct.unpack('2h', struct.pack('I', header[49]))[0]
            opt['em_wavelength'] = [em_wavelength]
            
            # MultiSIM001采集的数据，  timepoint包括step，phase不包括step
            # MultiSIM002采集的数据，  timepoint不包括step，phase包括step
            # MultiSIM003采集的数据，  timepoint不包括step，phase不包括step
            # 为区别，利用脚本将源数据文件头修改为：2step数据的timepoints包括step，3step数据的timepoints不包括step，phase均不包括step
            if opt['num_phase'] == 10:
                opt['num_phase'] = 5
                opt['num_step'] = 2
                opt['num_timepoint'] = header[45] // 65536 # 超过65536会解析错误,甚至不清楚会溢出到哪里！
            elif '-3step' in basename or '-2step' in basename:
                opt['num_timepoint'] = header[45] // 65536 # 超过65536会解析错误,甚至不清楚会溢出到哪里！
            else:
                opt['num_timepoint'] = header[45] // 65536 // opt['num_step'] # 超过65536会解析错误,甚至不清楚会溢出到哪里！
            # numbers of time points
            if not self.is_SIM_rawdata and opt['num_timepoint'] <= 0: opt['num_timepoint'] = 1 # OTF

            if '405' in basename:
                opt['ex_wavelength'] = [405]
            elif '445' in basename:
                opt['ex_wavelength'] = [445]
            elif '488' in basename:
                opt['ex_wavelength'] = [488]
            elif '561' in basename:
                opt['ex_wavelength'] = [561]
            elif '640' in basename:
                opt['ex_wavelength'] = [640]
            else:
                # raise RuntimeError
                opt['ex_wavelength'] = [-1]


            try_depth = header[24] % 65536
            if try_depth > 1:
                opt['num_pixel_depth'] = try_depth
                assert_tps = opt['num_section'] // (opt['num_phase'] * opt['num_orientation'] * opt['num_pixel_depth'] * opt['num_channel'] * opt['num_step'])
                if opt['num_timepoint'] == assert_tps:
                    pass
                elif opt['num_timepoint'] == assert_tps + 1:
                    opt['num_timepoint'] -= 1  # labview's bug
                else:
                    print('header may be wrong')
            else:
                if opt['num_timepoint'] == 0:
                    opt['num_pixel_depth'] = 0
                else:
                    temp = opt['num_section'] // (opt['num_phase'] * opt['num_orientation'] * opt['num_timepoint'] * opt['num_channel'] * opt['num_step'])
                    if temp == 0:
                        opt['num_timepoint'] = opt['num_timepoint'] - 1
                        if opt['num_timepoint'] == 0:
                            opt['num_pixel_depth'] = 0
                        else:
                            opt['num_pixel_depth'] = opt['num_section'] // (opt['num_phase'] * opt['num_orientation'] * opt['num_timepoint'] * opt['num_channel'] * opt['num_step'])
                    else:
                        opt['num_pixel_depth'] = temp

            if not self.is_SIM_rawdata and opt['num_pixel_depth'] > 3 and opt['num_pixel_depth'] == opt['num_pixel_width']: opt['num_pixel_depth'] = 3 # OTF

            # data save mode. Z: depth | W: number of wave | T: timepoints
            # opt['data_save_order'] = {'0': 'ZTW', '1': 'WZT', '2': 'ZWT'}[str(header[45] % 65536)]  # header[45]中另一位超过65536后溢出到哪里？
            opt['data_save_order'] = 'ZTW'


        # additional option:
        # pixel per xx
        opt['pixel_per_timepoint'] = opt['num_pixel_width'] * opt['num_pixel_height'] * opt['num_pixel_depth'] * opt['num_phase'] * opt['num_orientation'] * opt['num_channel'] * \
                                     opt['num_step']
        opt['pixel_per_slice'] = opt['num_pixel_width'] * opt['num_pixel_height'] * opt['num_phase'] * opt['num_orientation'] * opt['num_channel'] * opt['num_step']
        opt['pixel_per_rawslice'] = opt['num_pixel_width'] * opt['num_pixel_height']
        # order
        opt['num_order'] = ((1 + opt['num_phase']) // 2)
        # freq sampling (in 1/um)
        opt['width_freq_sampling'] = 1.e19 if opt['width_space_sampling'] == 0 else 1 / (opt['num_pixel_width'] * opt['width_space_sampling'])
        opt['height_freq_sampling'] = 1.e19 if opt['height_space_sampling'] == 0 else 1 / (opt['num_pixel_height'] * opt['height_space_sampling'])
        opt['depth_freq_sampling'] = 1.e19 if opt['depth_space_sampling'] * opt['num_pixel_depth'] == 0 else 1 / (opt['num_pixel_depth'] * opt['depth_space_sampling'])

        # dxy dkr
        opt['radial_space_sampling'] = max(opt['width_space_sampling'], opt['height_space_sampling'])
        opt['radial_freq_sampling'] = min(opt['width_freq_sampling'], opt['height_freq_sampling'])

        data_length = os.path.getsize(self.file) - 1024
        if data_length == opt['num_pixel_height'] * opt['num_pixel_width'] * opt['num_section'] * opt['byte_per_pixel']:
            pass
        elif data_length <= opt['num_pixel_height'] * opt['num_pixel_width'] * opt['num_section'] * opt['byte_per_pixel']:
            # if data_length % (opt['num_pixel_height'] * opt['num_pixel_width'] * opt['byte_per_pixel'] * opt['num_phase'] * opt['num_orientation'] *
            #                   opt['num_pixel_depth'] * opt['num_channel'] * opt['num_step']) == 0:
            opt['num_section'] = data_length // (opt['num_pixel_height'] * opt['num_pixel_width'] * opt['byte_per_pixel'])
            opt['num_timepoint'] = opt['num_section'] // (opt['num_phase'] * opt['num_orientation'] * opt['num_pixel_depth'] * opt['num_channel'] * opt['num_step'])
            print('mrc header is corrupted and this program has auto-reconvered it. please check the mrc header')
        else:
            print('mrc header is corrupted and we cannot fix it. please check the mrc header')

        return opt

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # def update(self):
    #
    #     opt = self.opt
    #
    #     if opt['num_pixel_depth'] == 0:
    #         header, _ = self.__read_mrc_header(self.big_endian)
    #         if header[45] % 65536 // opt['num_step'] > 1: # tps > 1
    #             max(1, int(opt['num_section'] // (opt['num_phase'] * opt['num_orientation'], * opt['num_timepoint'] * opt['num_channel'] * opt['num_step'])))
    #
    #     self.totalsize = os.path.getsize(self.file)
    #
    #     self.opt['num_section'] = int((self.totalsize - HEADER_SIZE) // (opt['num_pixel_width'] * opt['num_pixel_height'] * opt['byte_per_pixel']))
    #
    #     self.opt['num_timepoint'] = int(self.opt['num_section'] // (
    #                 max(1, opt['num_pixel_depth']) * opt['num_phase'] * opt['num_orientation'] * opt['num_channel'] * opt['num_step']))


    # ----------------------------------------
    # get total binary data
    # ----------------------------------------
    def __get_total_data(self):

        # self.update()

        pixel_read = self.opt['num_pixel_width'] * self.opt['num_pixel_height'] * self.opt['num_section']

        if self.opt['is_complex']: pixel_read *= 2

        with open(self.file, 'rb') as f:
            raw_image = np.fromfile(f, dtype=self.big_endian_signal + self.opt['dtype_symbol'], count=pixel_read, offset=HEADER_SIZE)

        if self.opt['is_complex']: raw_image = raw_image[0::2] + 1j * raw_image[1::2]

        # with open(self.file, 'rb') as f:
        #     f.seek(HEADER_SIZE)
        #     raw_image = struct.unpack(self.big_endian_signal + str(pixel_read) + self.opt['dtype_symbol'], f.read(pixel_read * self.opt['byte_per_pixel']))

        return raw_image

    # ----------------------------------------
    # get binary data at given timepoints
    # ----------------------------------------
    def __get_timepoint_data(self, begin_timepoint, read_timepoint):

        # self.update()

        assert begin_timepoint + read_timepoint <= self.opt['num_timepoint'], 'out of mrc length'

        pixel_pass = begin_timepoint * self.opt['pixel_per_timepoint']

        pixel_read = read_timepoint * self.opt['pixel_per_timepoint']

        if self.opt['is_complex']:
            pixel_pass *= 2
            pixel_read *= 2

        with open(self.file, 'rb') as f:
            raw_image = np.fromfile(f, dtype=self.big_endian_signal + self.opt['dtype_symbol'], count=pixel_read, offset=HEADER_SIZE + pixel_pass * self.opt['byte_per_pixel'])

        # with open(self.file, 'rb') as f:
        #     f.seek(HEADER_SIZE + pixel_pass * self.opt['byte_per_pixel'])
        #     raw_image = struct.unpack(self.big_endian_signal + str(pixel_read) + self.opt['dtype_symbol'], f.read(pixel_read * self.opt['byte_per_pixel']))

        return raw_image

    # ----------------------------------------
    # get binary data at given timepoints, get depth
    # ----------------------------------------
    def __get_slice_data(self, timepoint, depth):

        # self.update()

        assert timepoint < self.opt['num_timepoint'], 'out of mrc length'

        pixel_pass = timepoint * self.opt['pixel_per_timepoint'] + depth * self.opt['pixel_per_slice']

        pixel_read = self.opt['pixel_per_slice']

        if self.opt['is_complex']:
            pixel_pass *= 2
            pixel_read *= 2

        with open(self.file, 'rb') as f:
            raw_image = np.fromfile(f, dtype=self.big_endian_signal + self.opt['dtype_symbol'], count=pixel_read, offset=HEADER_SIZE + pixel_pass * self.opt['byte_per_pixel'])

        # with open(self.file, 'rb') as f:
        #     f.seek(HEADER_SIZE + pixel_pass * self.opt['byte_per_pixel'])
        #     raw_image = struct.unpack(self.big_endian_signal + str(pixel_read) + self.opt['dtype_symbol'], f.read(pixel_read * self.opt['byte_per_pixel']))

        return raw_image

    # ----------------------------------------
    # get binary data at given timepoints, get depth
    # ----------------------------------------
    def __get_rawslice_data(self, slice_begin, slice_num):

        # self.update()

        assert (slice_begin + slice_num) < self.opt['num_timepoint'] * self.opt['num_orientation'] * self.opt['num_phase'] * self.opt['num_channel'] * self.opt['num_pixel_depth'], 'out of mrc length'

        pixel_pass = slice_begin * self.opt['pixel_per_rawslice']

        pixel_read = slice_num * self.opt['pixel_per_rawslice']

        if self.opt['is_complex']:
            pixel_pass *= 2
            pixel_read *= 2

        with open(self.file, 'rb') as f:
            raw_image = np.fromfile(f, dtype=self.big_endian_signal + self.opt['dtype_symbol'], count=pixel_read, offset=HEADER_SIZE + pixel_pass * self.opt['byte_per_pixel'])

        # with open(self.file, 'rb') as f:
        #     f.seek(HEADER_SIZE + pixel_pass * self.opt['byte_per_pixel'])
        #     raw_image = struct.unpack(self.big_endian_signal + str(pixel_read) + self.opt['dtype_symbol'], f.read(pixel_read * self.opt['byte_per_pixel']))

        return raw_image

    # ----------------------------------------
    # convert to [T, O, P, C, D, H, W] or [T, S, O, P, C, D, H, W]
    # ----------------------------------------
    def __convert_dtype(self, x, convert_to_tensor=True, do_reshape=True, convert_to_float32=True):

        assert self.opt['num_channel'] == 1, "chennel of the raw data should be 1"

        if convert_to_tensor: # as_float应该放到reshape后return前,其它同
            if self.opt['is_complex']:
                x = torch.from_numpy(x.astype(np.complex64))
            else:
                if convert_to_float32:
                    x = torch.from_numpy(x.astype(np.float32))
                else:
                    x = torch.from_numpy(x)
        else:
            if convert_to_float32:
                if self.opt['is_complex']:
                    pass
                else:
                    x = x.astype(np.float32)

        if do_reshape:

            if self.opt['num_step'] == 1:

                (W, H, D, O, P, C) = (self.opt['num_pixel_width'], self.opt['num_pixel_height'], self.opt['num_pixel_depth'],
                                      self.opt['num_orientation'], self.opt['num_phase'], self.opt['num_channel'])
                T = int(x.squeeze().shape[0] / (W * H * D * O * P * C))

                fn = torch.Tensor.permute if convert_to_tensor else np.transpose

                if self.opt['data_save_order'] == 'ZTW':  # [C, T, D, O, P, H, W] -> [T, O, P, C, D, H, W]
                    x = fn(x.reshape(C, T, D, O, P, H, W), (1, 3, 4, 0, 2, 5, 6))
                elif self.opt['data_save_order'] == 'WZT':  # [T, D, C, O, P, H, W] -> [T, O, P, C, D, H, W]
                    x = fn(x.reshape(T, D, C, O, P, H, W), (0, 3, 4, 2, 1, 5, 6))
                elif self.opt['data_save_order'] == 'ZWT':  # [T, C, D, O, P, H, W] -> [T, O, P, C, D, H, W]
                    x = fn(x.reshape(T, C, D, O, P, H, W), (0, 3, 4, 1, 2, 5, 6))
                else:
                    raise NotImplementedError("check the code!")

            elif self.opt['num_step'] >= 2:

                (W, H, D, O, P, S, C) = (self.opt['num_pixel_width'], self.opt['num_pixel_height'], self.opt['num_pixel_depth'],
                                         self.opt['num_orientation'], self.opt['num_phase'], self.opt['num_step'], self.opt['num_channel'])
                T = int(x.squeeze().shape[0] / (W * H * D * O * P * S * C))

                fn = torch.Tensor.permute if convert_to_tensor else np.transpose

                if self.opt['data_save_order'] == 'ZTW':  # [C, T, D, O, P, S, H, W] -> [T, S, O, P, C, D, H, W]
                    x = fn(x.reshape(C, T, D, O, P, S, H, W), (1, 5, 3, 4, 0, 2, 6, 7))
                elif self.opt['data_save_order'] == 'WZT':  # [T, D, C, O, P, S, H, W] -> [T, S, O, P, C, D, H, W]
                    x = fn(x.reshape(T, D, C, O, P, S, H, W), (0, 5, 3, 4, 2, 1, 6, 7))
                elif self.opt['data_save_order'] == 'ZWT':  # [T, C, D, O, P, S, H, W] -> [T, S, O, P, C, D, H, W]
                    x = fn(x.reshape(T, C, D, O, P, S, H, W), (0, 5, 3, 4, 1, 2, 6, 7))
                else:
                    raise NotImplementedError("check the code!")

            else:
                raise NotImplementedError("check the code!")

        return x

    # ----------------------------------------
    # convert to [T, O, P, C, D, H, W], T=D=1
    # ----------------------------------------
    def __convert_dtype_from_slice(self, x, convert_to_tensor=True, do_reshape=True, convert_to_float32=True):

        assert self.opt['num_channel'] == 1, "chennel of the raw data should be 1"

        if convert_to_tensor: # as_float应该放到reshape后return前,其它同
            if self.opt['is_complex']:
                x = torch.from_numpy(x.astype(np.complex64))
            else:
                if convert_to_float32:
                    x = torch.from_numpy(x.astype(np.float32))
                else:
                    x = torch.from_numpy(x)
        else:
            if convert_to_float32:
                if self.opt['is_complex']:
                    pass
                else:
                    x = x.astype(np.float32)

        if do_reshape:

            if self.opt['num_step'] == 1:

                (W, H, O, P) = (self.opt['num_pixel_width'], self.opt['num_pixel_height'], self.opt['num_orientation'], self.opt['num_phase'])
                D, T, C = 1, 1, 1

                fn = torch.Tensor.permute if convert_to_tensor else np.transpose

                if self.opt['data_save_order'] == 'ZTW':  # [C, T, D, O, P, H, W] -> [T, O, P, C, D, H, W]
                    x = fn(x.reshape(C, T, D, O, P, H, W), (1, 3, 4, 0, 2, 5, 6))
                elif self.opt['data_save_order'] == 'WZT':  # [T, D, C, O, P, H, W] -> [T, O, P, C, D, H, W]
                    x = fn(x.reshape(T, D, C, O, P, H, W), (0, 3, 4, 2, 1, 5, 6))
                elif self.opt['data_save_order'] == 'ZWT':  # [T, C, D, O, P, H, W] -> [T, O, P, C, D, H, W]
                    x = fn(x.reshape(T, C, D, O, P, H, W), (0, 3, 4, 1, 2, 5, 6))
                else:
                    raise NotImplementedError("check the code!")

            elif self.opt['num_step'] >= 2:

                (W, H, O, P, S) = (self.opt['num_pixel_width'], self.opt['num_pixel_height'], self.opt['num_orientation'], self.opt['num_phase'], self.opt['num_step'])

                D, T, C = 1, 1, 1

                fn = torch.Tensor.permute if convert_to_tensor else np.transpose

                if self.opt['data_save_order'] == 'ZTW':  # [C, T, D, O, P, S, H, W] -> [T, S, O, P, C, D, H, W]
                    x = fn(x.reshape(C, T, D, O, P, S, H, W), (1, 5, 3, 4, 0, 2, 6, 7))
                elif self.opt['data_save_order'] == 'WZT':  # [T, D, C, O, P, S, H, W] -> [T, S, O, P, C, D, H, W]
                    x = fn(x.reshape(T, D, C, O, P, S, H, W), (0, 5, 3, 4, 2, 1, 6, 7))
                elif self.opt['data_save_order'] == 'ZWT':  # [T, C, D, O, P, S, H, W] -> [T, S, O, P, C, D, H, W]
                    x = fn(x.reshape(T, C, D, O, P, S, H, W), (0, 5, 3, 4, 1, 2, 6, 7))
                else:
                    raise NotImplementedError("check the code!")

            else:
                raise NotImplementedError("check the code!")

        return x

    # ----------------------------------------
    # convert to [H, W]
    # ----------------------------------------
    def __convert_dtype_from_rawslice(self, x, convert_to_tensor=True, convert_to_float32=True):
        if convert_to_tensor: # as_float应该放到reshape后return前,其它同
            if self.opt['is_complex']:
                x = torch.from_numpy(x.astype(np.complex64))
            else:
                if convert_to_float32:
                    x = torch.from_numpy(x.astype(np.float32))
                else:
                    x = torch.from_numpy(x)
        else:
            if convert_to_float32:
                if self.opt['is_complex']:
                    pass
                else:
                    x = x.astype(np.float32)
        (W, H) = (self.opt['num_pixel_width'], self.opt['num_pixel_height'])
        return x.reshape(-1, W, H)

    # ----------------------------------------
    # get total data as torch mat
    # ----------------------------------------
    def get_total_data_as_mat(self, convert_to_tensor=True, do_reshape=True, convert_to_float32=True):
        x = self.__get_total_data()
        x = self.__convert_dtype(x, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape, convert_to_float32=convert_to_float32)
        return x

    # ----------------------------------------
    # get data at given timepoints as torch mat
    # ----------------------------------------
    def get_timepoint_data_as_mat(self, begin_timepoint, read_timepoint, strict=False, convert_to_tensor=True, do_reshape=True, convert_to_float32=True):
        if read_timepoint == 0 or begin_timepoint >= self.opt['num_timepoint']:
            return None
        elif begin_timepoint + read_timepoint > self.opt['num_timepoint']:
            if strict: return None
            read_timepoint = self.opt['num_timepoint'] - begin_timepoint

        x = self.__get_timepoint_data(begin_timepoint, read_timepoint)
        x = self.__convert_dtype(x, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape, convert_to_float32=convert_to_float32)
        return x

    # ----------------------------------------
    # get data at a given slice as torch mat
    # ----------------------------------------
    def get_slice_data_as_mat(self, timepoint, depth, convert_to_tensor=True, do_reshape=True, convert_to_float32=True):
        if timepoint >= self.opt['num_timepoint']:
            x = None
        else:
            x = self.__get_slice_data(timepoint, depth)
            x = self.__convert_dtype_from_slice(x, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape, convert_to_float32=convert_to_float32)
        return x

    # ----------------------------------------
    # get data at a given raw slice as torch mat
    # ----------------------------------------
    def get_rawslice_data_as_mat(self, slicebegin, slicenum, convert_to_tensor=True, convert_to_float32=True):
        x = self.__get_rawslice_data(slicebegin, slicenum)
        x = self.__convert_dtype_from_rawslice(x, convert_to_tensor=convert_to_tensor, convert_to_float32=convert_to_float32)
        return x

    # ----------------------------------------
    # get next timepoint (batch)
    # ----------------------------------------
    def get_next_timepoint_batch(self, batchsize=1, convert_to_tensor=True, do_reshape=True, strict=False, convert_to_float32=True):
        # self.update()
        begin_timepoint = self.timepoint_have_been_read
        read_timepoint = batchsize

        if begin_timepoint + read_timepoint <= self.opt['num_timepoint']:
            self.timepoint_have_been_read += batchsize
            return self.get_timepoint_data_as_mat(begin_timepoint, read_timepoint, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape, convert_to_float32=convert_to_float32)
        else:
            if strict is False:  # 1 < read < batchsize
                timepoint_can_be_read = self.opt['num_timepoint'] - begin_timepoint
                if timepoint_can_be_read != 0:
                    self.timepoint_have_been_read += timepoint_can_be_read
                    return self.get_timepoint_data_as_mat(begin_timepoint, timepoint_can_be_read, convert_to_tensor=convert_to_tensor, do_reshape=do_reshape, convert_to_float32=convert_to_float32)
                else:
                    return None
            else:
                return None

    def turning_back(self, tps): # 回溯
        self.timepoint_have_been_read = max(0, self.timepoint_have_been_read - tps)


# ----------------------------------------
# Class: WriteMRC
# ----------------------------------------
# Method:
#       -> __init__ make opt and header for mrc file to be writed from opt
#       -> write header
#       -> [api] write data in append mode
# [safe]:
#   do not deliver pointer (handle)
#   control pointer scope with 'with'
# ----------------------------------------

class WriteMRC:
    # ----------------------------------------
    # __init__ method (structor function in c++)
    # ----------------------------------------
    def __init__(self, file, header, big_endian=False, compress=False):
        # assert big_endian is False  # 不知道C程序又把header哪里写错了，不能调用原有header改为大端
        # if compress == 'uint8norm' and big_endian:
        #     print('the saved mrc {} (uint8 & big endian) can not be loaded by imageJ (bio-format)'.format(file))
        # 存为大端自动调用imagej的bio-format, 存为小端则不自动调用.
        # bio-format虽然能读出z和t，但是（1）整体读取较慢（2）读正在写入的数据可能会出错
        assert compress in [# 不压缩
                            None, False,
                            # 不对灰度值做scale处理，+表示将小于零的数置为零
                            'float32', 'float32+', 'uint16', 'int16', 'uint8', 'complex64',
                            # float32norm/uint16norm将前1%的均值置为1.0/1000，从而平衡bleach
                            'float32norm', 'uint16norm',
                            # # uint8norm将前10个像素的均值置为255，无法很好的平衡bleach
                            # 'uint8norm',
                            # 不要使用16位浮点，除非重写bio-format插件
                            'float16', 'float16+', 'float16norm'
                            ]
        self.compress = compress
        self.file = file
        self.header = header
        self.big_endian = big_endian
        self.header = correct_header(self.header, big_endian=big_endian)
        self.init_header()
        if self.compress: self.make_compress_header()
        self.header = tuple(self.header)
        self.opt = self.get_option_from_mrc_header()
        self.big_endian_signal = '>' if self.big_endian else '<'
        self.write_mrc_header()
        self.timepoint_have_been_written = 0

    def get_option_from_mrc_header(self):

        opt = dict2nonedict({})
        header = self.header

        basename = os.path.basename(self.file).lower()

        opt['num_pixel_height'] = header[1]
        opt['num_pixel_width'] = header[0]

        if '-3step' in basename:
            opt['num_step'] = 3
        elif '-2step' in basename:
            opt['num_step'] = 2
        else:
            temp = header[57] // 65536
            opt['num_step'] = temp if temp in [1, 2] else 1

        opt['is_complex'], opt['byte_per_pixel'], opt['dtype_symbol'] = \
            {'6': [False, 2, 'H'], '4': [True, 8, 'f'], '2': [False, 4, 'f'], '1': [False, 2, 'i'], '0': [False, 1, 'B'], '12': [False, 2, 'e']}[str(header[3])]

        if self.big_endian is False:
            opt['num_orientation'] = header[41] // 65536
            opt['num_phase'] = header[41] % 65536
            # opt['num_channel'] = max(1, struct.unpack('2h', struct.pack('I', header[49]))[0])
            opt['num_channel'] = 1
            opt['num_pixel_depth'] = max(1, header[24] // 65536)
            # data saving mode, Z: depth | W: number of wave | T: timepoints
            # opt['data_save_order'] = {'0': 'ZTW', '1': 'WZT', '2': 'ZWT'}[str(header[45] // 65536)]
            opt['data_save_order'] = 'ZTW' # header[45]中另一位超过65536后溢出到这边
        else:
            opt['num_orientation'] = header[41] % 65536
            opt['num_phase'] = header[41] // 65536
            # opt['num_channel'] = max(1, struct.unpack('2h', struct.pack('I', header[49]))[1])
            opt['num_channel'] = 1
            opt['num_pixel_depth'] = max(1, header[24] % 65536)
            # data saving mode, Z: depth | W: number of wave | T: timepoints
            # opt['data_save_order'] = {'0': 'ZTW', '1': 'WZT', '2': 'ZWT'}[str(header[45] % 65536)] # header[45]中另一位超过65536后溢出到哪里？
            opt['data_save_order'] = 'ZTW'

        return opt

    def write_mrc_header(self):
        with open(self.file, 'wb') as f:
            for idx in range(int(HEADER_SIZE // 4)):
                f.write(struct.pack(self.big_endian_signal + 'I', self.header[idx]))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    def update(self):
        if self.big_endian is False:
            opt = self.opt
            header = list(self.header)
            self.opt['num_timepoint'] = self.timepoint_have_been_written
            self.opt['num_section'] = opt['num_timepoint'] * opt['num_pixel_depth'] * opt['num_phase'] * opt['num_orientation'] * opt['num_channel'] * opt['num_step']
            header[2] = self.opt['num_section']
            # print(self.opt['num_timepoint'])
            
            # MultiSIM001采集的数据，  timepoint包括step，phase不包括step
            # MultiSIM002采集的数据，  timepoint不包括step，phase包括step
            # MultiSIM003采集的数据，  timepoint不包括step，phase不包括step
            # 为区别，利用脚本将源数据文件头修改为：2step数据的timepoints包括step，3step数据的timepoints不包括step，phase均不包括step
            basename = os.path.basename(self.file).lower()
            if '-3step' in basename or '-2step' in basename:
                # header[45] = header[45] // 65536 + self.opt['num_timepoint'] * 1 # 超过65536帧溢出
                header[45] = self.opt['num_timepoint'] * 1
            else:
                # header[45] = header[45] // 65536 + self.opt['num_timepoint'] * self.opt['num_step'] # 超过65536帧溢出
                header[45] = self.opt['num_timepoint'] * self.opt['num_step']
            # header[9] = self.opt['num_timepoint']
            # print(header[45])
            self.header = tuple(header)
            with open(self.file, 'rb+') as f:
                for idx in range(int(HEADER_SIZE // 4)):
                    f.write(struct.pack(self.big_endian_signal + 'I', self.header[idx]))
        else:
            opt = self.opt
            header = list(self.header)
            self.opt['num_timepoint'] = self.timepoint_have_been_written
            self.opt['num_section'] = opt['num_timepoint'] * opt['num_pixel_depth'] * opt['num_phase'] * opt['num_orientation'] * opt['num_channel'] * opt['num_step']
            header[2] = self.opt['num_section']
            # print(self.opt['num_timepoint'])
            # MultiSIM001采集的数据，  timepoint包括step，phase不包括step
            # MultiSIM002采集的数据，  timepoint不包括step，phase包括step
            # MultiSIM003采集的数据，  timepoint不包括step，phase不包括step
            # 为区别，利用脚本将源数据文件头修改为：2step数据的timepoints包括step，3step数据的timepoints不包括step，phase均不包括step
            basename = os.path.basename(self.file).lower()
            if '-3step' in basename or '-2step' in basename:
                header[45] = header[45] % 65536 + self.opt['num_timepoint'] * 1 * 65536 # 超过65536帧鬼知道会怎么溢出
            else:
                header[45] = header[45] % 65536 + self.opt['num_timepoint'] * self.opt['num_step'] * 65536 # 超过65536帧鬼知道会怎么溢出
            # header[9] = self.opt['num_timepoint']
            # print(header[45])
            self.header = tuple(header)
            with open(self.file, 'rb+') as f:
                for idx in range(int(HEADER_SIZE // 4)):
                    f.write(struct.pack(self.big_endian_signal + 'I', self.header[idx]))

    def init_header(self):
        if isinstance(self.header, tuple): self.header = list(self.header)
        self.header[2] = 0
        if self.big_endian is False:
            # self.header[45] = self.header[45] // 65536 # 超过65536帧溢出
            pass
        else:
            self.header[45] = self.header[45] % 65536 # 超过65536帧鬼知道会怎么溢出

    # ----------------------------------------
    # write binary data
    # ----------------------------------------
    def write_otf_data(self, x):
        self.__write_mrc_append(x)

    def __write_mrc_append(self, x):
        assert len(x.shape) == 1, "check code!"
        save_data_type = {'6': 'uint16', '2': 'float32', '4': 'complex64', '1': 'int16', '0': 'uint8', '12':'float16'}[str(self.header[3])]

        if save_data_type == 'complex64':
            assert self.opt['is_complex'] and (not self.compress) # complex data can not be compressed
            x = x.astype(np.complex64)
            # This is a strange implement inapplicable to visualization. Anyway we follow it.
            x_complex = np.zeros(2 * x.shape[0], dtype=np.float32)
            x_complex[0::2] = np.real(x)
            x_complex[1::2] = np.imag(x)
            x = x_complex
            x = x.astype(self.big_endian_signal + 'f')
        # elif dtype_symbol == 'e':
        #     x = x.astype(self.big_endian_signal+'e')
        elif save_data_type in ['float32']:
            x = x.astype(self.big_endian_signal + 'f')
        elif save_data_type in ['float16']:
            x[x > 65504] = 65504
            x[x < -65504] = -65504
            x = x.astype(np.float16) # auto clip
        elif save_data_type == 'uint16':
            x[x < 0] = 0
            x[x > 65535] = 65535
            x = x.astype(self.big_endian_signal + 'H')
        elif save_data_type == 'int16':
            x[x < -32768] = -32768
            x[x > 32767] = 32767
            x = x.astype(self.big_endian_signal + 'i')
        elif save_data_type == 'uint8':
            x[x < 0] = 0
            x[x > 255] = 255
            x = x.astype(self.big_endian_signal + 'B')
        else:
            raise RuntimeError('the dtype of stack to be saved must be uint16, float32, or complex64')

        with open(self.file, 'ab') as f:
            temp = x.tobytes()
            # f.write(data.tobytes()) | x.tofile(f) # very slow
            endian = 'big' if self.big_endian else 'little'
            b = ba.bitarray(endian=endian)
            b.frombytes(temp)
            b.tofile(f)

    # ----------------------------------------
    # from torch [T, O, P, C, D, H, W] convert to binary
    # ----------------------------------------
    def __convert_dtype(self, x):

        if self.compress in ['float32+', 'float16+', 'float32norm', 'float16norm', 'uint16norm']: # 'uint8norm'
            x[x<0] = 0

        if self.compress in ['float32norm', 'float16norm', 'uint16norm']:
            resultant_norm_intensity = RESULTANT_NORM_INTENSITY if self.compress == 'uint16norm' else 1.0  # [500,2000] for [2d, 3d]
            if len(x.shape) == 7:
                for idx_T in range(x.shape[0]):
                    for idx_C in range(x.shape[3]):
                        if isinstance(x, torch.Tensor):
                            temp = x[idx_T, :, :, idx_C, ...]  # TOPCDHW -> OPDHW
                            O, P, D, H, W = temp.shape
                            temp, _ = torch.max(temp.reshape(O * P * D, H, W), dim=0, keepdim=False)  # OPDHW -> HW
                            temp = temp[..., 20:-20, 20:-20].flatten()
                            scale = torch.topk(temp, temp.shape[0] // 100, sorted=False)[0].mean()
                            if scale > 0:
                                x[idx_T, :, :, idx_C, ...] = x[idx_T, :, :, idx_C, ...] / (scale / resultant_norm_intensity)
                            else:
                                x[idx_T, :, :, idx_C, ...] = torch.zeros_like(x[idx_T, :, :, idx_C, ...])
                        else:
                            temp = np.max(x[idx_T, :, :, idx_C, ...], axis=(0, 1, 2), keepdims=False).squeeze()  # TOPCDHW -> OPDHW -> HW
                            temp = temp[20:-20, 20:-20].flatten()
                            scale = temp[np.argpartition(temp, - temp.shape[0] // 100)[-temp.shape[0] // 100:]].mean()
                            if scale > 0:
                                x[idx_T, :, :, idx_C, ...] = x[idx_T, :, :, idx_C, ...] / (scale / resultant_norm_intensity)
                            else:
                                x[idx_T, :, :, idx_C, ...] = np.zeros_like(x[idx_T, :, :, idx_C, ...])
            else:
                for idx_T in range(x.shape[0]):
                    for idx_S in range(x.shape[1]):
                        for idx_C in range(x.shape[4]):
                            if isinstance(x, torch.Tensor):
                                temp = x[idx_T, idx_S, :, :, idx_C, ...]  # TSOPCDHW -> OPDHW
                                O, P, D, H, W = temp.shape
                                temp, _ = torch.max(temp.reshape(O * P * D, H, W), dim=0, keepdim=False)  # OPDHW -> HW
                                temp = temp[..., 20:-20, 20:-20].flatten()
                                scale = torch.topk(temp, temp.shape[0] // 100, sorted=False)[0].mean()
                                if scale > 0:
                                    x[idx_T, idx_S, :, :, idx_C, ...] = x[idx_T, idx_S, :, :, idx_C, ...] / (scale / resultant_norm_intensity)
                                else:
                                    x[idx_T, idx_S, :, :, idx_C, ...] = torch.zeros_like(x[idx_T, idx_S, :, :, idx_C, ...])
                            else:
                                temp = np.max(x[idx_T, idx_S, :, :, idx_C, ...].astype(np.float32), axis=(0, 1, 2), keepdims=False).squeeze()  # TSOPCDHW -> OPDHW -> HW
                                temp = temp[20:-20, 20:-20].flatten()
                                scale = temp[np.argpartition(temp, - temp.shape[0] // 100)[-temp.shape[0] // 100:]].mean()
                                if scale > 0:
                                    x[idx_T, idx_S, :, :, idx_C, ...] = x[idx_T, idx_S, :, :, idx_C, ...] / (scale / resultant_norm_intensity)
                                else:
                                    x[idx_T, idx_S, :, :, idx_C, ...] = np.zeros_like(x[idx_T, idx_S, :, :, idx_C, ...])

        # elif self.compress in ['uint8norm']:
        #     if isinstance(x, torch.Tensor): x = x.cpu().numpy()
        #     if len(x.shape) == 7:
        #         for idx_T in range(x.shape[0]):
        #             for idx_C in range(x.shape[3]):
        #                 temp = np.max(x[idx_T, :, :, idx_C, ...].astype(np.float32), axis=(0, 1, 2), keepdims=False).squeeze()  # TOPCDHW -> OPDHW -> HW
        #                 temp = temp[20:-20, 20:-20].flatten()
        #                 scale = np.mean(temp[np.argpartition(temp, - 10)[-10:]])
        #                 if scale > 0:
        #                     x[idx_T, :, :, idx_C, ...] = x[idx_T, :, :, idx_C, ...] / (scale / 255.0)
        #                 else:
        #                     x[idx_T, :, :, idx_C, ...] = np.zeros_like(x[idx_T, :, :, idx_C, ...])
        #     else:
        #         for idx_T in range(x.shape[0]):
        #             for idx_S in range(x.shape[1]):
        #                 for idx_C in range(x.shape[4]):
        #                     temp = np.max(x[idx_T, idx_S, :, :, idx_C, ...].astype(np.float32), axis=(0, 1, 2), keepdims=False).squeeze()  # TSOPCDHW -> OPDHW -> HW
        #                     temp = temp[20:-20, 20:-20].flatten()
        #                     scale = np.mean(temp[np.argpartition(temp, - 10)[-10:]])
        #                     if scale > 0:
        #                         x[idx_T, idx_S, :, :, idx_C, ...] = x[idx_T, idx_S, :, :, idx_C, ...] / (scale / 255.0)
        #                     else:
        #                         x[idx_T, idx_S, :, :, idx_C, ...] = np.zeros_like(x[idx_T, idx_S, :, :, idx_C, ...])

        opt = self.opt

        if isinstance(x, torch.Tensor):
            from_tensor = True
            fn = torch.Tensor.permute
        elif isinstance(x, np.ndarray):
            from_tensor = False
            fn = np.transpose
        else:
            raise NotImplementedError('check the code')

        if len(x.shape) == 7:
            assert opt['num_step'] == 1
        elif len(x.shape) == 8:
            assert opt['num_step'] >= 2
        else:
            raise NotImplementedError('check the code')

        if opt['num_step'] == 1:

            (T, O, P, C, D, H, W) = x.shape

            assert (O, P, C, D, H, W) == (
                opt['num_orientation'], opt['num_phase'], opt['num_channel'], opt['num_pixel_depth'], opt['num_pixel_height'], opt['num_pixel_width']), \
                "wrong data size {} and {}".format(
                    [opt['num_orientation'], opt['num_phase'], opt['num_channel'], opt['num_pixel_depth'], opt['num_pixel_height'], opt['num_pixel_width']],
                    [O, P, C, D, H, W])

            assert C == opt['num_channel']

            if opt['data_save_order'] == 'ZTW':  # [T, O, P, C, D, H, W] -> [C, T, D, O, P, H, W]
                x = fn(x, (3, 0, 4, 1, 2, 5, 6)).reshape(-1)
            elif opt['data_save_order'] == 'WZT':  # [T, O, P, C, D, H, W] -> [T, D, C, O, P, H, W]
                x = fn(x, (0, 4, 3, 1, 2, 5, 6)).reshape(-1)
            elif opt['data_save_order'] == 'ZWT':  # [T, O, P, C, D, H, W] -> [T, C, D, O, P, H, W]
                x = fn(x, (0, 3, 4, 1, 2, 5, 6)).reshape(-1)
            else:
                raise NotImplementedError("check the code")

        else: # opt['num_step'] >= 2

            (T, S, O, P, C, D, H, W) = x.shape

            assert (S, O, P, C, D, H, W) == (
                opt['num_step'], opt['num_orientation'], opt['num_phase'], opt['num_channel'], opt['num_pixel_depth'], opt['num_pixel_height'], opt['num_pixel_width']
            ), "wrong data size {} and {}".format([S, O, P, C, D, H, W],
                                                  [opt['num_step'], opt['num_orientation'], opt['num_phase'], opt['num_channel'], opt['num_pixel_depth'], opt['num_pixel_height'],
                                                   opt['num_pixel_width']])

            assert C == opt['num_channel']

            if opt['data_save_order'] == 'ZTW':  # [T-0, S-1, O-2, P-3, C-4, D-5, H-6, W-7] -> [C, T, D, O, P, S, H, W]
                x = fn(x, (4, 0, 5, 2, 3, 1, 6, 7)).reshape(-1)
            elif opt['data_save_order'] == 'WZT':  # [T, S, O, P, C, D, H, W] -> [T, D, C, O, P, S, H, W]
                x = fn(x, (0, 5, 4, 2, 3, 1, 6, 7)).reshape(-1)
            elif opt['data_save_order'] == 'ZWT':  # [T, S, O, P, C, D, H, W] -> [T, C, D, O, P, S, H, W]
                x = fn(x, (0, 4, 5, 2, 3, 1, 6, 7)).reshape(-1)
            else:
                raise NotImplementedError("check the code")

        if from_tensor:
            x = x.cpu().numpy()

        return x, T

    # ----------------------------------------
    # write torch mat as binary in append mode
    # ----------------------------------------
    def write_data_append(self, x):
        x, T = self.__convert_dtype(x)
        self.__write_mrc_append(x)
        self.timepoint_have_been_written += T
        self.update()

    def make_compress_header(self):
        if isinstance(self.header, tuple): self.header = list(self.header)
        if self.compress in ['uint8']: # 'uint8norm'
            self.header[3] = 0
        elif self.compress == 'int16':
            self.header[3] = 1
        elif self.compress in ['uint16', 'uint16norm']:
            self.header[3] = 6
        elif self.compress in ['float16', 'float16+', 'float16norm']:
            self.header[3] = 12
        else:
            # default: self.header[3] = 2 or 1 [float32 / float32+ / float32norm or complex64]
            pass


def write_mrc_image(data, path, sampling_rate=None, datatype='single', big_endian=False):
    """
    Fast Write [without spacing/freq sampling rate] only for debug!
    """
    # assert big_endian is False # 暂不支持大端文件头
    header = [512, 512, 189, 6, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1119092736, 1119092736, 1119092736, 1, 2, 3, 0, 0, 0, 0, 0, 49312, 0, 0, 0, 0, 0, 0, 0,
              131072, 0, 0, 1176256512, 0, 1176256512, 0, 1176256512, 0, 196611, 7274496, 0, 1176256512, 21, 0, 0, 0, 1, 0, 0, 0, 0, 0, 10, 1819043171, 51, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1819043171, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    (T, O, P, C, D, H, W) = data.shape
    header[45] = 65536 * 0 + T  # data_save_order = 0 (ZTW), num_timepoint = T
    header[41] = 65536 * O + P  # orientation = O, phase = P
    header[24] = D * 65536 + 49312
    header[1] = H
    header[0] = W
    header[2] = T * O * P * C * D
    if datatype == 'single':
        header[3] = 2  # 2=single, 6=uint16
    elif datatype == 'uint16':
        header[3] = 6  # 2=single, 6=uint16
    else:
        raise NotImplementedError
    if sampling_rate is not None:
        header[10] = struct.unpack('I', struct.pack('f', sampling_rate[2]))[0]  # W
        header[11] = struct.unpack('I', struct.pack('f', sampling_rate[1]))[0]  # H
        header[12] = struct.unpack('I', struct.pack('f', sampling_rate[0]))[0]  # D
    wm = WriteMRC(path, header, big_endian=big_endian)
    wm.write_data_append(data)