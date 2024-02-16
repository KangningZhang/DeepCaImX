clc
clear
close all
tic
[file,path] = uigetfile('*.tiff');
%% Settings
N_start = 1; % Number of the first frame loaded
N_end = 400; % Number of the last frame loaded
FOV_x = 64; % Height of the FOV in pixel after tiling
FOV_y = 64; % Width of the FOV in pixel after tiling
nt = 100; % Length of the sub-video in frame after tiling
Overlapping_x = 11; % Overlapping length on height in pixel when tiling
Overlapping_y = 11; % Overlapping length on weight in pixel when tiling
normalization_option = true; % Whether need normalization to range of 0~1 after tiling. (default=true)
rgb2gray_option = false; % Whether need to convert RGB to grayscale, as the tailor only process grayscale tiff images. (default=false)
name_output = 'data'; % Name of output data
data_format = 'tiff'; % Choose the data format of output. Choices: 'tiff', 'mat'
%% Loading
if rgb2gray_option == 1
    temp = single(rgb2gray(imread([path,file])));
else
    temp = single(imread([path,file]));
end
imgs = zeros(size(temp,1),size(temp,2),N_end-N_start+1);
for i = N_start : N_end
    if rgb2gray_option == 1
        imgs(:,:,i-N_start+1) = single(rgb2gray(imread([path,file],i)));
    else
        imgs(:,:,i-N_start+1) = single(imread([path,file],i));
    end
end
%% Tiling
n_x = floor((size(imgs,1)-Overlapping_x-1)/(FOV_x-Overlapping_x))+1;
n_y = floor((size(imgs,2)-Overlapping_y-1)/(FOV_y-Overlapping_y))+1;
n_t = floor((N_end-N_start)/nt) + 1;
imgs_padding = single(zeros((n_x-1)*(FOV_x-Overlapping_x)+FOV_x,(n_y-1)*(FOV_y-Overlapping_y)+FOV_y,n_t*nt));
imgs_padding(1:size(imgs,1),1:size(imgs,2),1:size(imgs,3)) = imgs;
f = waitbar(0, 'Starting');
for i = 1 : n_x*n_y*n_t
    waitbar(i/(n_x*n_y*n_t), f, sprintf('Progress: %d %%', floor(i/(n_x*n_y*n_t)*100)));
    i_x = mod((i-1),n_x);
    i_y = mod(floor((i-1)/n_x),n_y);
    i_t = floor((i-1)/n_x/n_y);
    imgs_tile = single(squeeze(imgs_padding(i_x*(FOV_x-Overlapping_x)+1:i_x*(FOV_x-Overlapping_x)+FOV_x,...
        i_y*(FOV_y-Overlapping_y)+1:i_y*(FOV_y-Overlapping_y)+FOV_y, i_t*nt+1:i_t*nt+nt)));
    if normalization_option == 1
            imgs_tile = (imgs_tile - min(imgs_tile(:))) / (max(imgs_tile(:)) - min(imgs_tile(:)));
    end
    if strcmp(data_format,'mat')
    save([name_output,'_',num2str(i),'.mat'], 'imgs_tile'); % save sub-video in a format of .mat
    elseif strcmp(data_format,'tiff')
        for k = 1 : size(imgs_tile,3)
            imwrite(double(squeeze(imgs_tile(:,:,k))),[name_output,'_',num2str(i),'.tiff'],"WriteMode","append"); % save sub-video in a format of .tiff
        end
    end
end
close(f)
toc