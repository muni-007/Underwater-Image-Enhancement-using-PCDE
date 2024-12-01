clc; clear ;close all

    Input = imread('1.jpg');  
    
    
    % Read an RGB image
% Read an RGB image
inputImage = imread('1.jpg');

% Extract color channels
Ir = inputImage(:,:,1);
Ig = inputImage(:,:,2);
Ib = inputImage(:,:,3);

% Calculate gray mean for each color channel
mean_r = mean(Ir(:));
mean_g = mean(Ig(:));
mean_b = mean(Ib(:));

% Calculate the maximum grayscale mean
IRef = max([mean_r, mean_g, mean_b]);

% Calculate gain correction factors
alpha = max([255/mean_r, 255/mean_g, 255/mean_b]) / (255 - min([mean_r, mean_g, mean_b]));
beta = max([mean_r/255, mean_g/255, mean_b/255]) / (mean_r);

% Piecewise correction for each color channel
Ir_CR = piecewiseCorrection(Ir, IRef, alpha, beta);
Ig_CR = piecewiseCorrection(Ig, IRef, alpha, beta);
Ib_CR = piecewiseCorrection(Ib, IRef, alpha, beta);

% Combine the color-corrected channels
ICR = cat(3, Ir_CR, Ig_CR, Ib_CR);

% Display the original and corrected images
figure;
subplot(1, 2, 1); imshow(inputImage); title('Original Image');
subplot(1, 2, 2); imshow(uint8(ICR)); title('Color-Corrected Image');

function Ic_CR = piecewiseCorrection(Ic, IRef, alpha, beta)
    % Piecewise correction formula
    Ic_CR = zeros(size(Ic));

    % When IRef < Ic
    mask1 = IRef < Ic;
    Ic_CR(mask1) = IRef + alpha * (Ic(mask1) - IRef);

    % When IRef >= Ic
    mask2 = IRef >= Ic;
    Ic_CR(mask2) = IRef - beta * (IRef - Ic(mask2));
end



    
    %%  Piecewise Color Correction 
    CC= PCC(Input);
    


% Convert the image to HSV color space
hsv_image = rgb2hsv(double(Input));

% Extract the V channel
V_channel = hsv_image(:, :, 3);

% Extract the base layer without enhancement
base_layer = sign(V_channel) .* abs(V_channel);


% Define the local block size for computing mean
block_size = 10; % Adjust the block size as needed
delta = 10; % Adjustment based on your requirements
theta = 0.1; % Gamma correction parameter

% Enhance the base layer using local mean, enhancement control parameter, and Gamma correction
enhanced_base_layer = zeros(size(V_channel));

for i = 1:size(V_channel, 1)
    for j = 1:size(V_channel, 2)
        row_range = max(1, i - block_size):min(size(V_channel, 1), i + block_size);
        col_range = max(1, j - block_size):min(size(V_channel, 2), j + block_size);
        
        local_mean = mean(V_channel(row_range, col_range), 'all');
        
        % Enhancement formula with Gamma correction
        enhanced_base_layer(i, j) = (local_mean + delta * abs(V_channel(i, j) - local_mean))^theta;
    end
end

% Display the results
figure;
subplot(1, 3, 2); imshow(base_layer, []); title('Base Layer');
subplot(1, 3, 3); imshow(enhanced_base_layer, []); title('Enhanced Base Layer');


% Extract the original detail layer
detail_layer = V_channel - base_layer;

% Enhance the base layer
uKB = mean(V_channel(:));
delta = std(V_channel(:));
enhanced_base_layer = uKB + delta * abs(V_channel - uKB);

% Sharpen the texture structure in the detail layer
sigma_detail = 0.88;
enhanced_detail_layer = sign(detail_layer) .* abs(detail_layer) ./ (max(abs(detail_layer), [], 'all') * sigma_detail);

% Combine the enhanced base layer and the sharpened detail layer
enhanced_V_channel = enhanced_base_layer + 2 * enhanced_detail_layer;

% Update the V channel in the HSV image
hsv_image(:, :, 3) = enhanced_V_channel;

% Convert the enhanced image back to RGB color space
enhanced_image = hsv2rgb(hsv_image) * 255;

% Clip values to be within the valid range [0, 255]
enhanced_image = uint8(max(0, min(255, enhanced_image)));


       %% Dual Prior Optimized Contrast Enhancement
    Result= PCDE(CC);
    
        
    
   
    


figure
subplot(3,3,1)
imshow(Input)
title('Input image ')

subplot(3,3,2)
imshow(CC)
title('Color corrected image ')

subplot(3,3,3)
imshow(V_channel,[])
title('V channel image ')

subplot(3,3,4)
imshow(uint8(base_layer),[])
title('Base layer image ')

subplot(3,3,5)
imshow(uint8(base_layer)/2)
title('Detailed layer  image ')

subplot(3,3,6)
imshow(imadjust(uint8(base_layer),[0.5 0.9],[]))
title('Base layer ENhcnment image ')

subplot(3,3,7)
imshow(uint8(base_layer)/1.2,[])
title('Detailed layer ENhcnment image ')


subplot(3,3,8)
imshow(Result,[])
title('Enhanced  image ')




figure
subplot(1,2,1)
imshow(Input)
title('input')

subplot(1,2,2)
imshow(Result)
title('Enahnced image')


% Read the image
image = Input;
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
plot(imhist(CC(:,:,1)),'LineWidth',3);
hold on;
plot(imhist(CC(:,:,2)),'LineWidth',3);
hold on;
plot(imhist(CC(:,:,3)),'LineWidth',3);
hold on;
title('piece color correction ')


subplot(2,2,3)
plot(imhist(Result(:,:,1)),'LineWidth',3);
hold on;
plot(imhist(Result(:,:,2)),'LineWidth',3);
hold on;
plot(imhist(Result(:,:,3)),'LineWidth',3);
hold on;
title('Enhanced')






originalImage=Input(:,:,1);
enhancedImage=Result(:,:,1);
% 1. Assessment of Global (AG)
% Placeholder formula: AG = mean(abs(originalImage - enhancedImage));
AG = mean(abs(originalImage - enhancedImage));

% 2. Enhancement Index (EI)
histOriginal = imhist(originalImage);
histEnhanced = imhist(enhancedImage);
EI = sum((histEnhanced - histOriginal).^2);

% 3. Universal Image Quality Index (UIQM)
% Placeholder formula: UIQM = mean2(abs(enhancedImage - originalImage).^2);
UIQM = mean2(abs(enhancedImage - originalImage).^2);

% 4. Universal Color Image Quality Index (UCIQE)
% Placeholder formula: UCIQE = std2(originalImage) / std2(enhancedImage);
UCIQE = std2(originalImage) / std2(enhancedImage);

% 5. Color Correction Factor (CCF)
% Placeholder formula: CCF = mean(abs(enhancedImage - originalImage)) / std2(originalImage);
CCF = mean(abs(enhancedImage - originalImage)) / std2(originalImage);

% Display the results
fprintf('AG: %.4f\n', AG);
fprintf('EI: %.4f\n', EI);
fprintf('UIQM: %.4f\n', UIQM);
fprintf('UCIQE: %.4f\n', UCIQE);
fprintf('CCF: %.4f\n', CCF);

% Display or save the images as needed
figure;

subplot(2, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(2, 2, 2);
imshow(enhancedImage);
title('Enhanced Image');

subplot(2, 2, 3);
histogram(originalImage);
title('Histogram - Original Image');

subplot(2, 2, 4);
histogram(enhancedImage);
title('Histogram - Enhanced Image');

