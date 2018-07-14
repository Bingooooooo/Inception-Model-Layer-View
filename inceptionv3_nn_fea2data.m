% Yanbing Jiang University of Waterloo Department of ECE
%% Access the trained model 
% clc;
clear all;
close all;

layer = 1;
output_col = [4 4 8 8 12];
output_row = [8 8 8 10 16];
net = inceptionv3(); 
%% See details of the architecture 
net.Layers 
% Read the image to classify 
% im = imread('TCM_outlier/goose-outlier_clear.jpg');
im = imread('C:\Users\y77jiang\Downloads\ILSVRC2012\set\ILSVRC2012_val_00000077.JPEG');
% map = hsv(256); % Or whatever colormap you want.
% im = ind2rgb(im, map); % im is a grayscale or indexed image.
% im = cat(3,im,im,im);
im = imresize(im,net.Layers(1).InputSize(1:2));
figure;
imshow(im);
% act_conv1 = activations(net,im,'conv2d_1','OutputAs','channels');
eval(['act_conv1 = activations(net,im,''conv2d_',num2str(layer),''',''OutputAs'',''channels'');']);
size(act_conv1)
act_conv1 = reshape(act_conv1,size(act_conv1,1),size(act_conv1,2),1,size(act_conv1,3));
act_scaled = mat2gray(act_conv1);
% figure;
% montage(act_scaled)

%%
tmp = act_scaled(:);
tmp = imadjust(tmp,stretchlim(tmp));
act_stretched = reshape(tmp,size(act_scaled));
figure;
% montage(act_stretched, 'Size', [4 8])
eval(['montage(act_stretched, ''Size'',[',num2str(output_col(layer)),' ',num2str(output_row(layer)),'])']);
% title('Activations from the conv2d_1 layer','Interpreter','none')
eval(['title(''Activations from the conv2d_',num2str(layer),' layer'',''Interpreter'',''none'')']);

%%
% act_bn1 = activations(net,im,'batch_normalization_1');
eval(['act_bn1 = activations(net,im,''batch_normalization_',num2str(layer),''');']);
act_bn1 = reshape(act_bn1,size(act_bn1,1),size(act_bn1,2),1,size(act_bn1,3));
act_scaled_2 = mat2gray(act_bn1);
% figure;
% montage(act_scaled)

%%
tmp2 = act_scaled_2(:);
tmp2 = imadjust(tmp2,stretchlim(tmp2));
act_stretched_2 = reshape(tmp2,size(act_scaled_2));
figure;
% montage(act_stretched_2, 'Size', [4 8])
eval(['montage(act_stretched_2, ''Size'',[',num2str(output_col(layer)),' ',num2str(output_row(layer)),'])']);
% title('Activations from the activation_1_BN layer','Interpreter','none')
eval(['title(''Activations from the activation_',num2str(layer),'_BN layer'',''Interpreter'',''none'')']);

%%
% act_relu1 = activations(net,im,'activation_1_relu');
eval(['act_relu1 = activations(net,im,''activation_',num2str(layer),'_relu'');']);
act_relu1 = reshape(act_relu1,size(act_relu1,1),size(act_relu1,2),1,size(act_relu1,3));
act_scaled_3 = mat2gray(act_relu1);
% figure;
% montage(act_scaled)

%%
tmp3 = act_scaled_3(:);
tmp3 = imadjust(tmp3,stretchlim(tmp3));
act_stretched_3 = reshape(tmp3,size(act_scaled_3));
figure;
% montage(act_stretched_3, 'Size', [4 8])
eval(['montage(act_stretched_3, ''Size'',[',num2str(output_col(layer)),' ',num2str(output_row(layer)),'])']);
% title('Activations from the activation_1_relu layer','Interpreter','none')
eval(['title(''Activations from the activation_',num2str(layer),'_ReLU layer'',''Interpreter'',''none'')']);

% dif = act_stretched - act_stretched_2;
% figure;
% montage(dif, 'Size', [4 8])
% title('Difference Between First Layer Before and after ReLu','Interpreter','none')

%%
% figure;
% subplot(1,2,1)
for i = 1:32
    eval(['a',num2str(i),'= act_stretched_3(:,:,:,',num2str(i),');']);
end
% title('Filter 1')
% subplot(1,2,2)
% imshow(act_stretched(:,:,:,2))
% title('Filter 2')

o1 = act_stretched(:,:,:,2);
w = net.Layers(3).Weights;
w1 = w(:,:,:,6);