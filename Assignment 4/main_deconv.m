clc
clear all
close all

%% READ IMAGES
lowNoise = double(imread('Blurred-LowNoise.png'));
medNoise = double(imread('Blurred-MedNoise.png'));
highNoise = double(imread('Blurred-HighNoise.png'));

load BlurKernel.mat
blurKernel = h;

%% INVERSE FILTER

invLowNoise = inverseFilter(lowNoise, blurKernel);
invMedNoise = inverseFilter(medNoise, blurKernel);
invHighNoise = inverseFilter(highNoise, blurKernel);

%% WEINER FILTER

wienerLowNoise = wienerFilter(lowNoise, blurKernel, 1);
wienerMedNoise = wienerFilter(medNoise, blurKernel, 5);
wienerHighNoise = wienerFilter(highNoise, blurKernel, 10);

%% PLOT :: INVERSE FILTER

figure()
subplot(1,3,1)
imshow(invLowNoise)
title("Inverse Filtering $\sigma = 1$", 'interpreter', 'latex')

subplot(1,3,2)
imshow(invMedNoise)
title("Inverse Filtering $\sigma = 5$", 'interpreter', 'latex')

subplot(1,3,3)
imshow(invHighNoise)
title("Inverse Filtering $\sigma = 10$", 'interpreter', 'latex')

%% PLOT :: WIENER FILTER

figure()
subplot(1,3,1)
% imshow(wienerLowNoise)
imshow(deconvwnr(lowNoise,blurKernel,1), [0,255])
title("Wiener Filtering $\sigma = 1$", 'interpreter', 'latex')

subplot(1,3,2)
imshow(100*wienerMedNoise, [0,255])
title("Wiener Filtering $\sigma = 5$", 'interpreter', 'latex')

subplot(1,3,3)
imshow(wienerHighNoise, [0,255])
title("Wiener Filtering $\sigma = 10$", 'interpreter', 'latex')

%% FUNCTION DEFINITIONS

function deconv = inverseFilter(image, kernel)
    [m,n] = size(image);
    
    fftImage = fft2(image, m,n);
    fftKernel = fft2(kernel, m,n);
    
    fftDeconv = fftImage./fftKernel;
    deconv = ifft2(fftDeconv);
end

function deconv = wienerFilter(image, kernel, noiseLevel)
    [m,n] = size(image);
    
    fftImage = fft2(image, m,n);
    fftKernel = fft2(kernel, m,n);
    
    fftDeconv = (conj(fftKernel)./(abs(fftKernel).^2 + noiseLevel^2)) * (fftImage);
    deconv = abs(ifft2(fftDeconv));
end