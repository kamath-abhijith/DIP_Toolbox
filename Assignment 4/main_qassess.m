clc
clear all
close all

%% LOAD DATA

load 'hw5/hw5.mat'
imageFiles = dir(fullfile('hw5/gblur/*.bmp'));
numImages = length(imageFiles);

imageFilenames = natsortfiles({imageFiles.name});

%% SCORES

psnrScore = zeros(1,145);
ssimScore = zeros(1,145);
for i = 1:145
    imageName = imageFiles(i).name;
    blurImage = imread(strcat('hw5/gblur/',string(imageFilenames(i))));
    trueImage = imread(strcat('hw5/refimgs/',string(refnames_blur(i))));

    mseScore = immse(blurImage, trueImage);
    psnrScore(i) = 10*log10(255/mseScore);
    [ssimScore(i), ~] = ssim(blurImage, trueImage);
end

%% CORRELATION

psnrCorr = corrcoef(psnrScore, blur_dmos(1:145))
ssimCorr = corrcoef(ssimScore, blur_dmos(1:145))
