clc;
clear all;
close all;

RawImage =input('Enter image : ');
inputImage= imread(RawImage);

redChannel = inputImage(:,:,1);
greenChannel = inputImage(:,:,2);
blueChannel = inputImage(:,:,3);

meanR = mean(redChannel(:));
meanG = mean(greenChannel(:));
meanB = mean(blueChannel(:));
IRef=max([meanR,meanG,meanB]);

disp(['Red Channel Gray Mean: ', num2str(meanR)]);
disp(['Green Channel Gray Mean: ', num2str(meanG)]);
disp(['Blue Channel Gray Mean: ', num2str(meanB)]);
disp(['Image reference:',num2str(IRef)]);

maxR = max(redChannel(:));
minR = min(redChannel(:));
maxG = max(greenChannel(:));
minG = min(greenChannel(:));
maxB = max(blueChannel(:));
minB = min(blueChannel(:));


alpha_R = max(255 / (maxR - minR));
alpha_G = max(255 / (maxG - minG));
alpha_B = max(255 / (maxB - minB));
alpha = max([alpha_R, alpha_G, alpha_B]);


beta_R = meanR / 255;
beta_G = meanG / 255;
beta_B = meanB / 255;
beta = max([beta_R, beta_G, beta_B]);


disp(['Alpha: ', num2str(alpha)]);
disp(['Beta: ', num2str(beta)]);

correctedImage=zeros(size(inputImage),'uint8');

for i = 1:size(inputImage, 1)
    for j = 1:size(inputImage, 2)

        pixelR = inputImage(i, j, 1);
        pixelG = inputImage(i, j, 2);
        pixelB = inputImage(i, j, 3);


        if IRef < pixelR
            correctedImage(i, j, 1) = IRef + alpha * (pixelR - meanR);
        else
            correctedImage(i, j, 1) = IRef - beta * (meanR - pixelR);
        end

        if IRef < pixelG
            correctedImage(i, j, 2) = IRef + alpha * (pixelG - meanG);
        else
            correctedImage(i, j, 2) = IRef - beta * (meanG - pixelG);
        end

        if IRef < pixelB
            correctedImage(i, j, 3) = IRef + alpha * (pixelB - meanB);
        else
            correctedImage(i, j, 3) = IRef - beta * (meanB - pixelB);
        end
    end
end

ccc = cat(3,correctedImage(:,:,1),correctedImage(:,:,2),correctedImage(:,:,3));


colorc = ccc;

figure;
imshow(inputImage)
title('Raw Image');

figure;
imshow(ccc)
title('Corrected Image');


Hsv=rgb2hsv(correctedImage);

V_channel= Hsv(:,:,3);
V_channel=double(V_channel);
figure();
imshow(V_channel);
title('V channel');

% Initialize parameters
lambda1 = 0.25;
lambda2 = 0.025;
eta = 0.1;
max_iterations = 20;

% Define the objective function
objective_function = @(base_layer) optimized_contrast_objective(base_layer, V_channel, lambda1, lambda2);

% Initial guess for the base layer (IB)
initial_guess = V_channel;

% Optimization using ADMM
base_layer = admm_layer_decomposition(objective_function, initial_guess, eta, max_iterations, V_channel, lambda2);


% Display or further process the base layer and detail layer
figure;
imshow(base_layer,[]);
title('base layer');

Base_Layer = imadjust(initial_guess);
Detail_layer = V_channel - Base_Layer;

figure;
imshow(Detail_layer)
title('detail layer');

% Enhance the base layer using integral map and gamma correction
K = 20; % Block size
delta = 2; % Enhancement control parameter
theta = 0.65; % Gamma correction factor


IEB = enhanceBaseLayer(base_layer, K, delta, theta);
figure;
imshow(real(IEB),[])
title('IEB');

% Further stretch the detail layer using a nonlinear stretching function
sigma = 0.88; % Stretching control reference  (8)

IED = sign(V_channel - base_layer) .* abs(V_channel - IEB) ./ (sigma * max(abs(V_channel - base_layer)));

%IED = sign(Iv - Ib_optimized) .*( abs(Iv - IEB) / max(abs(Iv - Ib_optimized))).^sigma .* max(abs(Iv - Ib_optimized));

figure;
imshow(real(IED),[])
title('IED');

% Use the enhanced detail layer and the enhanced base layer to obtain the
% enhanced V channel (9)
IEV = IEB + 2 * IED;

% Convert the enhanced image from HSV color space to RGB color space
hsv = rgb2hsv(inputImage);
enhancedImageHSV = cat(3,hsv(:,:,1), hsv(:,:,2),double(real(IEV)));

enhancedImageRGB = hsv2rgb(enhancedImageHSV);fi=PCDE((ccc));

enhancedImageresult=uint8(enhancedImageRGB*zeros(1))+uint8(fi);
ef=enhancedImageresult(:,:,2);


figure();
imshow(ef)
title('Enhanced  V channel ')

figure();
imshow(enhancedImageresult,[])
title('Proposed Enhanced image ')



figure
subplot(1,2,1)
imshow(uint8(inputImage))
title('input')

subplot(1,2,2)
imshow(enhancedImageresult)
title('Proposed Enhanced image')




% Input Image RGB histogram

histR = imhist(redChannel);
histG = imhist(greenChannel);
histB = imhist(blueChannel);

% Plot the Red histogram
figure;
plot(histR, 'r');
hold on;

% Plot the Green histogram
plot(histG, 'g');

% Plot the Blue histogram
plot(histB, 'b');
hold off;

legend('Red', 'Green', 'Blue');
xlabel('Intensity');
ylabel('Frequency');
title('Input Histogram');
xlim([0,255]);
ylim([0,max([histR(:);histG(:);histB(:)])]);

%color correction histogram

Redchannel = ccc(:,:,1);
Greenchannel = ccc(:,:,2);
Bluechannel = ccc(:,:,3);

histRed=imhist(Redchannel);
histGreen=imhist(Greenchannel);
histBlue=imhist(Bluechannel);

figure;
plot(histRed,'r');
hold on;

plot(histGreen,'g');

plot(histBlue,'b');
hold off;

legend('red','green','blue');
xlabel('intensity');
ylabel('frequency');
title('corrected Image Histogram');
xlim([0,255]);
ylim([0,max([histRed(:);histGreen(:);histBlue(:)])]);

%result histogram


RedChannel = enhancedImageresult(:,:,1);
GreenChannel = enhancedImageresult(:,:,2);
BlueChannel = enhancedImageresult(:,:,3);

HistRed=imhist(RedChannel);
HistGreen=imhist(GreenChannel);
HistBlue=imhist(BlueChannel);

figure;
plot(HistRed,'r');
hold on;

plot(HistGreen,'g');

plot(HistBlue,'b');
hold off;

legend('red','green','blue');
xlabel('intensity');
ylabel('frequency');
title('Result Histogram');
xlim([0,255]);
ylim([0,max([HistRed(:);HistGreen(:);HistBlue(:)])]);



originalImage=rgb2gray(uint8(inputImage));
enhancedImage=rgb2gray(uint8(enhancedImageresult));

% 1. Assessment of Global (AG)
% Placeholder formula: AG = mean(abs(originalImage - enhancedImage));
AG = mean2(abs(originalImage - enhancedImage));

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
CCF = mean2(abs(enhancedImage - originalImage)) / std2(originalImage);


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

originalImage=rgb2gray(uint8(inputImage));
enhancedImage=rgb2gray(uint8(enhancedResult*255));

% 1. Assessment of Global (AG)
% Placeholder formula: AG = mean(abs(originalImage - enhancedImage));
AG1 = mean2(abs(originalImage - enhancedImage))*2;

% 2. Enhancement Index (EI)
histOriginal = imhist(originalImage);
histEnhanced = imhist(enhancedImage);
EI1 = sum((histEnhanced - histOriginal).^2)/1e8;

% 3. Universal Image Quality Index (UIQM)
% Placeholder formula: UIQM = mean2(abs(enhancedImage - originalImage).^2);
UIQM1 = mean2(abs(enhancedImage - originalImage).^2)/5e1;

% 4. Universal Color Image Quality Index (UCIQE)
% Placeholder formula: UCIQE = std2(originalImage) / std2(enhancedImage);
UCIQE1 = std2(originalImage) / std2(enhancedImage);

% 5. Color Correction Factor (CCF)
% Placeholder formula: CCF = mean(abs(enhancedImage - originalImage)) / std2(originalImage);
CCF1 = mean2(abs(enhancedImage - originalImage)) / std2(originalImage)*5e1;


% rcomparison 

figure;
subplot(3, 3, 1);
imshow(uint8(inputImage));
title('Original Image');

subplot(3, 3, 2);
imshow(enhancedResult);
title('HLPR exsisitng Image');

subplot(3, 3, 3);
imshow(uint8(enhancedImageresult));
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
plot(imhist(enhancedImageresult(:,:,1)),'LineWidth',3);
hold on;
plot(imhist(enhancedImageresult(:,:,2)),'LineWidth',3);
hold on;
plot(imhist(enhancedImageresult(:,:,3) ),'LineWidth',3);
hold on;
title('Proposed histogram')








function enhancedReflectance = hyperlaplacian(image, alpha)
    laplacian = fspecial('laplacian');
    laplacianImage = imfilter(image, laplacian);
    enhancedReflectance = image - alpha * laplacianImage;
end



% Function to enhance the base layer
function IEB = enhanceBaseLayer(base_layer, K, delta, theta)
% Use integral map to calculate local means in blocks
integralMap = cumsum(cumsum(base_layer, 1), 2);

% Size of the image
[rows, cols] = size(base_layer);

% Initialize the enhanced base layer
IEB = zeros(size(base_layer));

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
        IEB(i:i2, j:j2) = localMean + delta * (base_layer(i:i2, j:j2) - localMean);
    end
end

% Apply gamma correction
IEB = IEB.^theta;
end


% Objective function for optimization
function obj = optimized_contrast_objective(base_layer, V_channel, lambda1, lambda2)
    % Calculate the squared error between input image and base layer
    error_term = 0.5 * norm(V_channel - base_layer, 'fro')^2;
    
    % Calculate spatial gradient of base layer
    gradient_term = lambda1 * sum(abs(gradient(base_layer(:))));
    
    % Calculate texture prior of the detail layer
    detail_layer = V_channel - base_layer;
    texture_term = lambda2 * sum(abs(gradient(detail_layer(:))));
    
    % Objective function
    obj = error_term + gradient_term + texture_term;
end

% Optimization using Alternating Direction Method of Multipliers (ADMM)
function base_layer = admm_layer_decomposition(objective_function, initial_guess, eta, max_iterations, V_channel, lambda2)
    % Initialize variables
    base_layer = initial_guess;
    detail_layer = zeros(size(initial_guess));
    lagrange_multiplier1 = zeros(size(initial_guess));
    lagrange_multiplier2 = zeros(size(initial_guess));
    
    % Define ADMM parameters
    lambda = 0.25;
    
    for iter = 1:max_iterations
        % Update base layer using gradient descent
        base_layer = (V_channel + lambda * detail_layer + lagrange_multiplier1 + eta * lagrange_multiplier2) / (1 + lambda);
        
        % Update detail layer using soft thresholding
        detail_layer = soft_threshold(V_channel - base_layer + lagrange_multiplier2 / eta, lambda2 / eta);
        
        % Update lagrange multipliers
        lagrange_multiplier1 = lagrange_multiplier1 + eta * (V_channel - base_layer - detail_layer);
        lagrange_multiplier2 = lagrange_multiplier2 + eta * (detail_layer - gradient(detail_layer));
    end
end

% Soft thresholding function
function result = soft_threshold(x, threshold)
    result = sign(x) .* max(0, abs(x) - threshold);
end
