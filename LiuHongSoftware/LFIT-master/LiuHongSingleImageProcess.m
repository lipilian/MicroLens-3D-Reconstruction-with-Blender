function result = LiuHongSingleImageProcess(OpticInfo, NumF)
    % ------------------------------------------------------
    % OpticInfo include all optic information and output resolution
    % information 
    % NumF which image user want to processed 
    % ------------------------------------------------------
    targetPath = OpticInfo.target_path;
    calibPath = OpticInfo.calibration_path;
    outputPath = OpticInfo.output_path;
    sensorType = OpticInfo.sensor_type;
    
    pixelPitch = OpticInfo.pixel_mm; % mm/n
    focLenMicro = OpticInfo.MLA_F_mm; % microlens array focal length in mm
    microPitch = OpticInfo.MLA_size_mm; % microlens pitch in mm
    sensorHight = OpticInfo.sensorH_resolution * pixelPitch; % 10 mm
    numMicroX = OpticInfo.num_Micro_X; % 80 lens along width
    numMicroY = OpticInfo.num_Micro_Y; % 80 lens along height
    dmin = OpticInfo.dmin_mm; % 20 / 20 / 20 mm actual 
    dmax = OpticInfo.dmax_mm;
    dnum = OpticInfo.dnum;
    dplane = linspace(dmin,dmax,dnum);