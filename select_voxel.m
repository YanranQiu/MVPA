%data = matfile("full_encoding_data.mat");

%sdata = data.ordmaps(:,1:30,:,:,:);

%%%%%%%%%%%%%% anova and calculate p values for each voxel %%%%%%%%%%%%%%
valence = {'v3','v1','v1','v1','v1','v1','v1','v3','v1','v1','v2','v2','v1','v1','v2','v2','v2','v1','v1','v2','v2','v3','v3','v2','v3','v3','v3','v3','v3','v3','v2','v2','v1','v1','v1','v1','v2','v1','v1','v1','v3','v3','v3','v3','v2','v2','v2','v2','v1','v2','v1','v1','v1','v2','v3','v3','v3','v3','v3','v3','v3','v3','v3','v3','v2','v2','v2','v2','v2','v2','v2','v2'};

pvalence = NaN(67,80,46);

for i = 1:67
    for j = 1:80
        for k = 1:46
            if ~isnan(sdata(1,1,i,j,k))
                tempdata = sdata(:,:,i,j,k);
                tempmatrix = tempdata';
                p = anova1(tempmatrix,valence,"off");
                pvalence(i,j,k) = p;
            end
        end
    end
end

save("pvalue-valence_select-voxel.mat","pvalence")

h1 = histogram(pvalence(:));
h1.BinWidth = 0.001;


%%%%%%%%%%%%%%%%%% select top 100 voxels %%%%%%%%%%%%%%%%%%

% load("pvalue-valence_select-voxel.mat");

min_index = ones(100,3);
min_value = ones(1,100);
cnt = 0;
excllist = zeros(2600,3);

while (sum(min_value==1)>0)
    currmin = find(pvalence==min(min(min(pvalence))));
    [a,b,c] = ind2sub(size(pvalence),currmin);
    tempp = pvalence(a,b,c);
    pvalence(a,b,c) = 1;
    if (~ismember([a,b,c],excllist,"rows"))
        cnt = cnt+1;
        min_index(cnt,:) = [a,b,c];
        min_value(1,cnt) = tempp;
        indi = (cnt-1)*26+1;
        indj = cnt*26;
        excllist(indi:indj,:) = [[a+1,b,c];[a-1,b,c];[a+1,b+1,c];[a-1,b+1,c];[a+1,b-1,c];[a-1,b-1,c];[a+1,b,c+1];[a-1,b,c+1];[a+1,b+1,c+1];[a-1,b+1,c+1];[a+1,b-1,c+1];[a-1,b-1,c+1];[a+1,b,c-1];[a-1,b,c-1];[a+1,b+1,c-1];[a-1,b+1,c-1];[a+1,b-1,c-1];[a-1,b-1,c-1];[a,b-1,c-1];[a,b+1,c-1];[a,b-1,c+1];[a,b+1,c+1];[a,b+1,c];[a,b-1,c];[a,b,c+1];[a,b,c-1]];
    end
end

save("top100voxels_valence.mat","min_index","min_value","excllist");
h2 = histogram(min_value);
h2.BinWidth = 0.00001;

%draw 3d plot of locations of voxels
x=min_index(:,1);
y=min_index(:,2);
z=min_index(:,3);
s=(normalize(min_value')+2)*20;
c=(normalize(min_value')+2)*20;
figure
scatter3(x,y,z,s,c,"filled")
view(30,35)

% load("pvalue-valence_select-voxel.mat");
% 
% min_index = ones(100,3);
% min_value = ones(1,100);
% cnt = 0;
% 
% while (sum(min_value==1)>0)
%     currmin = find(pvalence==min(min(min(pvalence))));
%     [a,b,c] = ind2sub(size(pvalence),currmin);
%     if (~ismember([a,b,c],excllist,"rows"))
%         cnt = cnt+1;
%         min_index(cnt,:) = [a,b,c];
%         min_value(1,cnt) = pvalence(a,b,c);
%         pvalence(a,b,c) = 1;
%         excllist = [excllist;[a+1,b,c];[a-1,b,c];[a+1,b+1,c];[a-1,b+1,c];[a+1,b-1,c];[a-1,b-1,c];[a+1,b,c+1];[a-1,b,c+1];[a+1,b+1,c+1];[a-1,b+1,c+1];[a+1,b-1,c+1];[a-1,b-1,c+1];[a+1,b,c-1];[a-1,b,c-1];[a+1,b+1,c-1];[a-1,b+1,c-1];[a+1,b-1,c-1];[a-1,b-1,c-1];[a,b-1,c-1];[a,b+1,c-1];[a,b-1,c+1];[a,b+1,c+1];[a,b+1,c];[a,b-1,c];[a,b,c+1];[a,b,c-1]];
%     end
% 
% end

% for i = 1:67
%     for j = 1:80
%         for k = 1:46
%             if ~isnan(pvalence(i,j,k))
%                 if min_value(,100)>pvalence(i,j,k)
%                     substitute;
%                     min_index.add(i,j,k);
%                     k = k + 3;
%                 end
%                 sort(min_value);
%             end
%         end
%     end
%     
% end

%%%%%%%%%%%%%%%%%%% transform training data for SVM %%%%%%%%%%%%%%%%%%%%%

data = matfile("full_encoding_data.mat");
trdata = data.ordmaps(:,31:70,:,:,:);

%load("top100voxels_valence.mat")

valence = [3,1,1,1,1,1,1,3,1,1,2,2,1,1,2,2,2,1,1,2,2,3,3,2,3,3,3,3,3,3,2,2,1,1,1,1,2,1,1,1,3,3,3,3,2,2,2,2,1,2,1,1,1,2,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2];
valence = valence';
writematrix(valence,"valence.csv");
train_data = zeros(2880,101);%72*40; 100+1

cnt = 1;
for i = 1:72
    for j = 1:40
        row = cnt;
        cnt = cnt+1;
        for k = 1:100
            a = min_index(k,1);
            b = min_index(k,2);
            c = min_index(k,3);
            train_data(row,k) = trdata(i,j,a,b,c);
        end
        train_data(row,101) = valence(i);
    end
end

% save("train_valence_top-voxel.mat","train_data");
writematrix(train_data,"train_valence_top-voxel_1.csv");
train_data_nonan = train_data( :, sum( isnan( train_data ) ) == 0);
writematrix(train_data_nonan,"train_valence_top-voxel_nonan.csv");
train_data_nonan_both = train_valence_top_voxel_1( :, sum( isnan( train_valence_top_voxel_1 ) ) == 0 & sum( isnan( test_data ) ) == 0);
writematrix(train_data_nonan_both,"train_valence_top-voxel_nonan_both.csv");

%%%%%%%%%%%%%%%%%%%%%%%% transform test data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = matfile("full_encoding_data.mat");
ttdata = data.ordmaps(:,71:100,:,:,:);

test_data = zeros(2160,101);%72*30; 100+1

cnt = 1;
for i = 1:72
    for j = 1:30
        row = cnt;
        cnt = cnt+1;
        for k = 1:100
            a = min_index(k,1);
            b = min_index(k,2);
            c = min_index(k,3);
            test_data(row,k) = ttdata(i,j,a,b,c);
        end
        test_data(row,101) = valence(i);
    end
end

% save("train_valence_top-voxel.mat","train_data");
writematrix(test_data,"test_valence_top-voxel.csv");
test_data_nonan = test_data( :, sum( isnan( train_valence_top_voxel_1 ) ) == 0); %take train data no nan, not test data
% any(isnan(test_data_nonan(:)));
writematrix(test_data_nonan,"test_valence_top-voxel_nonan_bytrain.csv");
test_data_nonan_ornot = test_data_nonan( :, sum( isnan( test_data_nonan ) ) == 0);
size(test_data_nonan_ornot)
test_data_nonan_both = test_data( :, sum( isnan( train_valence_top_voxel_1 ) ) == 0 & sum( isnan( test_data ) ) == 0);
writematrix(test_data_nonan_both,"test_valence_top-voxel_nonan_both.csv");