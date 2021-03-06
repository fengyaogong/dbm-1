%Most parts of this code were taken from Russlan S. 
%The code is reorganized and slightly modified

% Work with test files first 
fprintf(1,'You first need to download files:\n train-images-idx3-ubyte.gz\n train-labels-idx1-ubyte.gz\n t10k-images-idx3-ubyte.gz\n t10k-labels-idx1-ubyte.gz\n from http://yann.lecun.com/exdb/mnist/\n and gunzip them \n'); 

f = fopen('../data/t10k-images-idx3-ubyte','r');
[a,count] = fread(f,4,'int32');
  
g = fopen('../data/t10k-labels-idx1-ubyte','r');
[l,count] = fread(g,2,'int32');

fprintf(1,'Starting to convert Test MNIST images (prints 10 dots) \n'); 
n = 1000;

Df = cell(1,10);
for d=0:9,
	Df{d+1} = fopen(['test' num2str(d) '.ascii'],'w');
end;
  
for i=1:10,
	fprintf('.');
  	rawimages = fread(f,28*28*n,'uchar');
  	rawlabels = fread(g,n,'uchar');
  	rawimages = reshape(rawimages,28*28,n);

  	for j=1:n,
    		fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
    		fprintf(Df{rawlabels(j)+1},'\n');
  	end;
end;

fprintf(1,'\n');
for d=0:9,
	fclose(Df{d+1});
  	D = load(['test' num2str(d) '.ascii'],'-ascii');
  	fprintf('%5d Digits of class %d\n',size(D,1),d);
  	save(['test' num2str(d) '.mat'],'D','-mat');
end;

% Work with trainig files second  
f = fopen('../data/train-images-idx3-ubyte','r');
[a,count] = fread(f,4,'int32');

g = fopen('../data/train-labels-idx1-ubyte','r');
[l,count] = fread(g,2,'int32');

fprintf(1,'Starting to convert Training MNIST images (prints 60 dots)\n'); 
n = 1000;

Df = cell(1,10);
for d=0:9,
	Df{d+1} = fopen(['digit' num2str(d) '.ascii'],'w');
end;

for i=1:60,
	fprintf('.');
  	rawimages = fread(f,28*28*n,'uchar');
  	rawlabels = fread(g,n,'uchar');
  	rawimages = reshape(rawimages,28*28,n);
  	for j=1:n,
    		fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
    		fprintf(Df{rawlabels(j)+1},'\n');
  	end;
end;

fprintf(1,'\n');
for d=0:9,
	fclose(Df{d+1});
  	D = load(['digit' num2str(d) '.ascii'],'-ascii');
  	fprintf('%5d Digits of class %d\n',size(D,1),d);
  	save(['digit' num2str(d) '.mat'],'D','-mat');
end;

dos('rm *.ascii');
