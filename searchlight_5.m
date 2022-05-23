%67*80*46

% data = matfile("full_encoding_data.mat");
% sdata = data.ordmaps(:,1:30,:,:,:);
% trdata = data.ordmaps(:,31:70,:,:,:);
% ttdata = data.ordmaps(:,71:100,:,:,:);

% valence = [3,1,1,1,1,1,1,3,1,1,2,2,1,1,2,2,2,1,1,2,2,3,3,2,3,3,3,3,3,3,2,2,1,1,1,1,2,1,1,1,3,3,3,3,2,2,2,2,1,2,1,1,1,2,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2];
% load("arousal.csv");
% arousal = arousal';

% train_label = repmat(valence,1,70);
% test_label = repmat(valence,1,30);
train_label = repmat(arousal,1,70);
test_label = repmat(arousal,1,30);

accs = [1 1 1 1];

train_data = ones(5040,125);%72*70
test_data = ones(2160,125);%72*30

for i = 3:5:63
    for j = 3:5:78
        for k = 3:5:43
            if isnan(trdata(1,1,i,j,k))
                continue;
            end
            train_sdata = sdata(:,:,i-2:i+2,j-2:j+2,k-2:k+2);
            train_trdata = trdata(:,:,i-2:i+2,j-2:j+2,k-2:k+2);
            test_tdata = ttdata(:,:,i-2:i+2,j-2:j+2,k-2:k+2);
            if any(isnan(train_sdata(:))) || any(isnan(test_tdata(:))) || any(isnan(train_trdata(:)))
                continue;
            end
            
            row = 1;
            for b = 1:30
                for a = 1:72
                    temp_train = train_sdata(a,b,:,:,:);
                    train_data(row,:) = temp_train(:)';
                    row = row+1;
                end
            end
            for d = 1:40
                for c = 1:72
                    temp_train = train_trdata(c,d,:,:,:);
                    train_data(row,:) = temp_train(:)';
                    row = row+1;
                end
            end
            cnt = 1;
            for f = 1:30
                for e = 1:72
                    temp_test = test_tdata(e,f,:,:,:);
                    test_data(cnt,:) = temp_test(:)';
                    cnt = cnt+1;
                end
            end
            
            mdl = fitcecoc(train_data,train_label);
            yfit = predict(mdl,test_data);
            acc = (sum(yfit==test_label(:)))/2160;
            accs = [accs;acc i j k];
        end
     end
end


%vertcat(zeros(1,3), [a*(-2.75)+90.75, b*(2.75)-126.5,c*(4)-72]);

% save("searchlight_valence_accs_5.mat","accs");
% accs(1,:)=[];
% save("searchlight_valence_accs_5_correct.mat","accs");
% accs(1,:)=[];
% save("searchlight_arousal_accs_new.mat","accs");
save("searchlight_arousal_accs_5.mat","accs");
save("searchlight_arousal_accs_5_correct.mat","accs");
