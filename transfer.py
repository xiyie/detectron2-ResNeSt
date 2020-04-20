import pdb,os
import torch

source_file = "./temp"
source_file = os.path.join(source_file, 'model_final.pth')
target_file = "./temp"
target_file = os.path.join(target_file, 'new.pth')


model = torch.load(source_file)
weights_dict = model['model']
weights_name = list(weights_dict.keys())
#cur = weights_dict['backbone.bottom_up.res3.0.conv2.fc1.weight']

for each_weight_name in weights_name:
    if 'backbone' in each_weight_name \
            and 'fc' in each_weight_name \
            and 'weight' in each_weight_name:
        cur_weight = weights_dict[each_weight_name]
        print(each_weight_name,"   ", cur_weight.shape)

        new_weight = torch.unsqueeze(cur_weight, 2)
        new_weight = torch.unsqueeze(new_weight, 3)
        print(each_weight_name,"   ", new_weight.shape)

        weights_dict[each_weight_name] = new_weight

for each_weight_name in weights_name:
    if 'backbone' in each_weight_name \
            and 'conv2' in each_weight_name \
            and 'bn' in each_weight_name \
            and 'bn1' not in each_weight_name:
        print(each_weight_name)
        new_weight_name = each_weight_name.split('.') 
        new_weight_name[-2] += '0'
        new_weight_name = '.'.join(new_weight_name)

        weights_dict[new_weight_name] = weights_dict.pop(each_weight_name) 
        

print('----------------------')

weights_dict = model['model']
weights_name = list(weights_dict.keys())
for each_weight_name in weights_name:
    if 'backbone' in each_weight_name \
            and 'fc' in each_weight_name \
            and 'weight' in each_weight_name:
        cur_weight = weights_dict[each_weight_name]
        print(each_weight_name,"   ", cur_weight.shape)

 
for each_weight_name in weights_name:
    if 'backbone' in each_weight_name \
            and 'conv2' in each_weight_name \
            and 'bn' in each_weight_name \
            and 'bn1' not in each_weight_name:
        print(each_weight_name)


torch.save(model, target_file)
