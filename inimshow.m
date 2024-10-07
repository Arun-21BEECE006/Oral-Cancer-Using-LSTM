function J = inimshow(im,inim)
imSize = size(im);
inim = imresize(inim,imSize(1:2));
inim = normalizeImage(inim);
inim(inim<0.2) = 0;
cmap = jet(255).*linspace(0,1,255)';
inim = ind2rgb(uint8(inim*255),cmap)*255;

combinedImage = double(rgb2gray(im))/2 + inim;
combinedImage = normalizeImage(combinedImage)*255;
J = uint8(combinedImage);
imshow(uint8(combinedImage));
end

