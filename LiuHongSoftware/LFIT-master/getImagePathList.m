function imagePath = getImagePathList(path)
    imagePath = dir(fullfile(path, '*.tif'));
    imagePath = extractfield(imagePath, 'name');    
end 