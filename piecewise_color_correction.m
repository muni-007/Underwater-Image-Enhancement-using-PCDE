% Read the input image
input_image = double(imread('1.jpg'));

% Calculate the gray mean values for each color channel
[H, W, ~] = size(input_image);
gray_mean_values = mean(mean(input_image, 1), 2);

% Calculate the maximum grayscale mean from all color channels
IRef = max(gray_mean_values);

% Display the maximum grayscale mean value
disp(['Maximum Grayscale Mean (IRef): ' num2str(IRef)]);

% Define correction parameters
alpha_denominator = max(max(input_image, [], [1, 2]) - min(input_image, [], [1, 2]));
beta = max(gray_mean_values ./ 255);

% Avoid division by zero in the alpha calculation
alpha_denominator(alpha_denominator == 0) = eps;

% Calculate alpha
alpha = 255 / alpha_denominator;

% Initialize an array to store corrected channels
corrected_channels = zeros(size(input_image), 'like', input_image);

% Apply the ultimate correction for each color channel
for c = 1:3
    channel = input_image(:, :, c);

    % Calculate the ultimate correction
    mask1 = channel > gray_mean_values(c);
    corrected1 = IRef + alpha * (255 / alpha_denominator) * (channel - gray_mean_values(c));

    mask2 = ~mask1;
    corrected2 = IRef - beta(c) * (gray_mean_values(c) / 255) * (gray_mean_values(c) - channel);

    % Combine the two corrections
    corrected_channel = zeros(size(channel));
    corrected_channel(mask1) = corrected1(mask1);
    corrected_channel(mask2) = corrected2(mask2);

    % Store the corrected channel
    corrected_channels(:, :, c) = corrected_channel;
end

% Clip values to be within the valid range [0, 255]
corrected_image = uint8(max(0, min(255, corrected_channels)));

% Display the results
figure;
subplot(1, 2, 1); imshow(uint8(input_image)); title('Input Image');
subplot(1, 2, 2); imshow(corrected_image); title('Corrected Image');
