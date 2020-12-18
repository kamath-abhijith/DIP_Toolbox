clc
clear
close all

image = imread("images/lena.tif");
[m,n] = size(image);
smooth_image = smooth(image(:), 1);
smooth_image = reshape(image, m,n);

figure()
imshow(image)
figure()
imshow(smooth_image)

function denoise = smooth(x, delta)

N = length(x);
D = toeplitz([-1 zeros(1,N-1)], [-1 1 zeros(1,N-2)]);

denoise = (eye(N) + delta*D'*D)\x;

end