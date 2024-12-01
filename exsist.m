
clc
clear all
close all

inputImage = imread('1.jpg');  

% Convert to double for numerical calculations
inputImage = double(inputImage);

% Extract color channels
Ir = inputImage(:,:,1);
Ig = inputImage(:,:,2);
Ib = inputImage(:,:,3);

% Calculate the gray mean for each color channel
mean_r = sum(Ir(:)) / numel(Ir);
mean_g = sum(Ig(:)) / numel(Ig);
mean_b = sum(Ib(:)) / numel(Ib);


% Find the maximum grayscale mean
IRef = max([mean_r, mean_g, mean_b]);

% correction for each color channel
Ir_CR = mean(IRef) + max(255 ./ (max(Ir(:)) - min(Ir(:)))) * (Ir - mean(Ir));
Ig_CR = mean(IRef) + max(255 ./ (max(Ig(:)) - min(Ig(:)))) * (Ig - mean(Ig));
Ib_CR = mean(IRef) + max(255 ./ (max(Ib(:)) - min(Ib(:)))) * (Ib - mean(Ib));

ccc = cat(3, uint8(Ir_CR), uint8(Ig_CR), uint8(Ib_CR));

% Convert to HSV color space
hsvImage = rgb2hsv(ccc);

% Extract V channel (illumination)
V = hsvImage(:, :, 3);

% Separation of Illumination and Reflectance
illumination = V;
reflectance = inputImage ./ repmat(V, [1, 1, 3]);

% Enhance Illumination using Gaussian Priors
enhancedIllumination = imguidedfilter(illumination);

% Enhance Reflectance using Hyper-Laplacian Priors
alpha = 0.2; % Adjust the hyper-Laplacian parameter
enhancedReflectance = hyperlaplacian(reflectance, alpha);

% Combine Enhanced Illumination and Reflectance
enhancedImage = enhancedIllumination .* enhancedReflectance;

% Convert back to HSV color space
enhancedHSV = hsvImage;
enhancedHSV(:, :, 3) = enhancedIllumination;

% Convert back to RGB
enhancedResult = hsv2rgb(enhancedHSV);

% Display Results
figure;
subplot(2, 2, 1);
imshow(uint8(inputImage));
title('Original Image');

subplot(2, 2, 2);
imshow(enhancedResult);
title('Enhanced image');

function enhancedReflectance = hyperlaplacian(image, alpha)
    laplacian = fspecial('laplacian');
    laplacianImage = imfilter(image, laplacian);
    enhancedReflectance = image - alpha * laplacianImage;
end
