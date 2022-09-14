
class subst_model(torch.nn.Module):
    def __init__(self):
        super(subst_model, self).__init__()
        self.first_layer = torch.nn.Conv2D(3, 128, 8, 8, 0)
        self.otherlayers =...

    def forward(self, x, freq):
        x = self.first_layer(x) #N, 128, 4, 4
        x = torch.cat([x, freq]) #N, 131, 4, 4
        x = self.otherlayers(x)
        return x        
    
    
x.shape ==(N, 3, 32, 32) #Cifar10


class model(torch.nn.Module):
    def __init__(self):
        super()...
        self.first_layer = torch.nn.Conv2D(3, 128, 8, 8,0)
        self.otherlayers =...

    def forward(self, x, freq):
        x = self.first_layer(x) #N, 128, 4, 4
        x = torch.cat([x, freq]) #N, 131, 4, 4
        x = self.otherlayers(x)
        return x        
    
    
x.shape ==(N, 3, 32, 32) #Cifar10


