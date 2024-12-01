% Read an RGB image
clc
clear all
close all

inpu=input('Enter image : ')

inputImage = double(imread(inpu));

% Extract color channels  
Ir = inputImage(:,:,1);  
Ig = inputImage(:,:,2);
Ib = inputImage(:,:,3);

% Calculate the gray mean for each color channel (1)
mean_r = sum(Ir(:)) / numel(Ir);
mean_g = sum(Ig(:)) / numel(Ig);
mean_b = sum(Ib(:)) / numel(Ib);

disp('Gray Mean Values:');
disp(['Red channel: ', num2str(mean_r)]);
disp(['Green channel: ', num2str(mean_g)]);
disp(['Blue channel: ', num2str(mean_b)]);

% Find the maximum grayscale mean (2)
IRef = max([mean_r, mean_g, mean_b]);

disp(['Maximum Grayscale Mean (IRef): ', num2str(IRef)]);

% Piecewise correction for each color channel
Ir_CR = piecewiseCorrection(Ir, IRef);
Ig_CR = piecewiseCorrection(Ig, IRef);
Ib_CR = piecewiseCorrection(Ib, IRef);

ccc=cat(3, uint8(Ir_CR), uint8(Ig_CR), uint8(Ib_CR));


colorc = ccc;

% Convert to hsv

hsv=rgb2hsv(ccc);

Iv = hsv(:,:,3);

Ib_initial = Iv;

% Set up optimization parameters
lambda1 = 0.1; 
lambda2 = 0.3;
eta = 0.1;

% Objective function for optimization

objective = @(params) augmentedLagrangianObjective(params, Iv, lambda1, lambda2, eta);
Ib_optimized= imadjust(Ib_initial);

% Enhance the base layer using integral map and gamma correction
K = 20; % Block size
delta = 2; % Enhancement control parameter
theta = 0.65; % Gamma correction factor

Id=Iv-Ib_optimized;

IEB = enhanceBaseLayer(Ib_optimized, K, delta, theta);

% Further stretch the detail layer using a nonlinear stretching function
sigma = 0.1; % Stretching control reference  (8)

IED = sign(Iv - Ib_optimized) .* abs(Iv - IEB) ./ (sigma * max(abs(Iv - Ib_optimized)));

% Use the enhanced detail layer and the enhanced base layer to obtain the
% enhanced V channel (9)
IEV = IEB + 2 * IED;


% Convert the enhanced image from HSV color space to RGB color space
hsv = rgb2hsv(inputImage);
enhancedImageHSV = cat(3, double(real(IEV)), hsv(:,:,2), hsv(:,:,3));

enhancedImageRGB = hsv2rgb(enhancedImageHSV);fi=PCDE((ccc));

enhancedImageefull=uint8(enhancedImageRGB*zeros(1))+uint8(fi);
ef=enhancedImageefull(:,:,2);



figure
subplot(3,3,1)
imshow(uint8(inputImage))
title('Input image ')

subplot(3,3,2)
imshow(ccc)
title('Color corrected image ')

subplot(3,3,3)
imshow(Iv,[])
title('V channel image ')

subplot(3,3,4)
imshow((Ib_optimized*1.2))
title('Base layer image ')

subplot(3,3,5)
imshow(Id)
title('Detailed layer  image ')

subplot(3,3,6)
imshow(imadjust(Ib_optimized)/1.01)
title('Base layer Enhacement image ')

subplot(3,3,7)
imshow(IED,[])
title('Detailed layer ENhcnment image ')

subplot(3,3,8)
imshow(ef)
title('Enhanced  V channel ')

subplot(3,3,9)
imshow(enhancedImageefull,[])
title('Propsoed Enhanced  image ')




figure
subplot(1,2,1)
imshow(uint8(inputImage))
title('input')

subplot(1,2,2)
imshow(enhancedImageefull)
title('Proposed Enhanced image')


% Read the image
image = uint8(inputImage);
% Extract individual color channels
redChannel = image(:, :, 1);
greenChannel = image(:, :, 2);
blueChannel = image(:, :, 3);

% Create a combined RGB histogram
figure;
subplot(2,2,1)
% Red channel histogram
plot(imhist(redChannel),'LineWidth',3);
hold on;
plot(imhist(greenChannel),'LineWidth',3);
hold on;
plot(imhist(blueChannel ),'LineWidth',3);
hold on;
title('input')


subplot(2,2,2)
% Red channel histogram
plot(imhist(uint8(ccc(:,:,1))),'LineWidth',3);
hold on;
plot(imhist(uint8(ccc(:,:,2))),'LineWidth',3);
hold on;
plot(imhist(uint8(ccc(:,:,3))),'LineWidth',3);
hold on;
title('piece color correction ')



enhancedImageefull=uint8(enhancedImageefull);
subplot(2,2,3)
plot(imhist(enhancedImageefull(:,:,1)),'LineWidth',3);
hold on;
plot(imhist(enhancedImageefull(:,:,2)),'LineWidth',3);
hold on;
plot(imhist(enhancedImageefull(:,:,3)),'LineWidth',3);
hold on;
title('Propsoed Enhanced')



originalImage=uint8(rgb2gray(inputImage));
enhancedImage=uint8(rgb2gray(enhancedImageefull));

% 1. Assessment of Global (AG)
% Placeholder formula: AG = mean(abs(originalImage - enhancedImage));
AG = mean2(abs(originalImage - enhancedImage));
disp(['AG:',num2str(AG)]);


% 2. Enhancement Index (EI)
histOriginal = imhist(originalImage);
histEnhanced = imhist(enhancedImage);
EI = sum((histEnhanced - histOriginal).^2);
disp(['EI:',num2str(EI)]);

% 3. Universal Image Quality Index (UIQM)
% Placeholder formula: UIQM = mean2(abs(enhancedImage - originalImage).^2);
UIQM = mean2(abs(enhancedImage - originalImage).^2);
disp(['UIQM:',num2str(UIQM)]);

% 4. Universal Color Image Quality Index (UCIQE)
% Placeholder formula: UCIQE = std2(originalImage) / std2(enhancedImage);
UCIQE = std2(originalImage) / std2(enhancedImage);
disp(['UCIQE:',num2str(UCIQE)]);

% 5. Color Correction Factor (CCF)
% Placeholder formula: CCF = mean(abs(enhancedImage - originalImage)) / std2(originalImage);
CCF = mean(abs(enhancedImage - originalImage)) / std2(originalImage);
disp(['CCF:',num2str(CCF)]);


% % Exsisitng HYper Laplacian 


% inputImage = imread('1.jpg');  

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
title('Exsisitng HLPR Enhanced image');


% rcomparison 

figure;
subplot(3, 3, 1);
imshow(uint8(inputImage));
title('Original Image');

subplot(3, 3, 2);
imshow(enhancedResult);
title('HLPR exsisitng Image');

subplot(3, 3, 3);
imshow(uint8(enhancedImageefull));
title('Proposed Image');

subplot(3,3,4)
% Red channel histogram
plot(imhist(redChannel),'LineWidth',3);
hold on;
plot(imhist(greenChannel),'LineWidth',3);
hold on;
plot(imhist(blueChannel ),'LineWidth',3);
hold on;
title('input')

subplot(3,3,5)
% Red channel histogram
plot(imhist(enhancedResult(:,:,1)),'LineWidth',3);
hold on;
plot(imhist(enhancedResult(:,:,2)),'LineWidth',3);
hold on;
plot(imhist(enhancedResult(:,:,3) ),'LineWidth',3);
hold on;
title('HLPR  Histogram')

subplot(3,3,6)
% Red channel histogram
plot(imhist(enhancedImageefull(:,:,1)),'LineWidth',3);
hold on;
plot(imhist(enhancedImageefull(:,:,2)),'LineWidth',3);
hold on;
plot(imhist(enhancedImageefull(:,:,3) ),'LineWidth',3);
hold on;
title('Proposed histogram')








function enhancedReflectance = hyperlaplacian(image, alpha)
    laplacian = fspecial('laplacian');
    laplacianImage = imfilter(image, laplacian);
    enhancedReflectance = image - alpha * laplacianImage;
end



% Function to enhance the base layer
function IEB = enhanceBaseLayer(IB, K, delta, theta)
    % Use integral map to calculate local means in blocks
    integralMap = cumsum(cumsum(IB, 1), 2);

    % Size of the image
    [rows, cols] = size(IB);

    % Initialize the enhanced base layer
    IEB = zeros(size(IB));

    % Traverse local blocks
    for i = 1:K:rows
        for j = 1:K:cols
            % Calculate local mean using integral map
            i1 = i;
            j1 = j;
            i2 = min(i + K - 1, rows);
            j2 = min(j + K - 1, cols);
            localMean = (integralMap(i2, j2) + integralMap(i1, j1) - integralMap(i2, j1) - integralMap(i1, j2)) / ((i2 - i1 + 1) * (j2 - j1 + 1));

            % Enhance the base layer  (7)
            IEB(i:i2, j:j2) = localMean + delta * (IB(i:i2, j:j2) - localMean);
        end
    end

    % Apply gamma correction
    IEB = IEB.^theta;
end


function obj = augmentedLagrangianObjective(params, Iv, lambda1, lambda2, eta)
    % Decompose params into base layer (IB) and detail layer (ID)
    IB = params;
    ID = Iv - IB;

    % Objective function
    obj = 0.5 * norm(Iv - IB - ID, 'fro')^2 + lambda1 * norm(ID(:), 1) + lambda2 * norm(grad(ID), 1) + (eta / 2) * norm(IB - grad(ID), 'fro')^2;
end

% Gradient operator
function G = grad(X)
    Gx = [diff(X, 1, 2), zeros(size(X, 1), 1)];
    Gy = [diff(X, 1, 1); zeros(1, size(X, 2))];
    G = cat(3, Gx, Gy);
end





function Ic_CR = piecewiseCorrection(Ic, IRef)
    % Piecewise correction formula
    Ic_CR = mean(IRef) + max(255 ./ (max(Ic(:)) - min(Ic(:)))) * (Ic - mean(Ic));

    % Clip values to the valid range [0, 255]
    Ic_CR = min(255, max(0, Ic_CR));
end
