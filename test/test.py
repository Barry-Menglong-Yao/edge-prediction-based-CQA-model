import torch 
# a=torch.tensor([[[[0.9 , 0.04 ]],

#          [[0.06 , 0.4 ]]],


#         [[[0.3 , 0.3 ]],

#          [[0.5 , 0.9 ]]]])
a=torch.rand(2,1,2)
print(a.shape)
print(a)
# b =torch.tensor([[[[0.4 , 0.9 ],
#           [0.3, 0.2]]],


#         [[[0.5, 0.2],
#           [0.1, 0.3]]]])
b=torch.rand(2,2,2)
# b=torch.tensor([[[0.2 , 0.1 ] ],

#         [[0.3 , 0.4 ] ]])
print(b.shape)
print(b)

c=a+ b 
print(c.shape) 
print(c)