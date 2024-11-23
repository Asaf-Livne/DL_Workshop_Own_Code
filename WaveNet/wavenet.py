from imports import *


class WaveNetModel(nn.Module):
     def __init__(self,dilation_repeats, dilation_depth, num_channels, kernel_size):

         self.dilation_repeats = dilation_repeats
         self.dilation_depth = dilation_depth
         self.num_blocks = dilation_repeats * dilation_depth

         super(WaveNetModel, self).__init__()
         self.casual_conv = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1)
         self.end_conv_1 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
         self.end_conv_2 = nn.Conv1d(in_channels=num_channels, out_channels=1, kernel_size=1)
         self.linear_mixer = nn.Conv1d(in_channels = self.num_blocks*num_channels, out_channels=num_channels, kernel_size=1)
         # dilated conv
         self.gated_convs = nn.ModuleList()
         self.filtered_convs = nn.ModuleList()
         # 1*1 conv list for residual connection
         self.residual_convs = nn.ModuleList()
         self.skip_convs = nn.ModuleList()


         for repeat_num in range(dilation_repeats):
             layer_gated_convs = nn.ModuleList()
             layer_filtered_convs = nn.ModuleList()
             for depth_num in range(dilation_depth):
                 layer_gated_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, dilation=2**depth_num, padding = 0))
                 layer_filtered_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, dilation=2**depth_num, padding = 0))
                 self.residual_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=1))
                 self.skip_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=1))
             self.gated_convs.append(layer_gated_convs)
             self.filtered_convs.append(layer_filtered_convs)

     def wavenet_block(self, x, repeat_num, dilation_depth, block_num):
         residual = x.clone() 
         gate_x = x.clone()  
         filter_x = x.clone()  
         gate_x = self.gated_convs[repeat_num][dilation_depth](gate_x)
         filter_x = self.filtered_convs[repeat_num][dilation_depth](filter_x)
         # Apply activation functions
         filter = torch.tanh(filter_x)
         gate = torch.sigmoid(gate_x)
         new_x = filter * gate
         skip = new_x
         res_conv = self.residual_convs[block_num](new_x)
         residual = residual [:, :, -res_conv.size(2):]
         residual += res_conv
         return skip, residual


     def wavenet_end(self, skip_sum, output_size):
         skip_sum = torch.cat([s[:, :, -output_size :] for s in skip_sum], dim=1)
         x = self.linear_mixer(skip_sum)
         x = F.relu(x)
         x = self.end_conv_1(x)
         x = F.relu(x)
         x = self.end_conv_2(x)
         return x


     def forward(self, input):
         x = self.casual_conv(input)
         skip_sum = []
         for block_num in range(self.num_blocks):
             repeat_num = block_num // self.dilation_depth
             dilation_depth = block_num % self.dilation_depth
             skip, residual = self.wavenet_block(x, repeat_num, dilation_depth, block_num)
             x = residual
             skip_sum.append(skip)
         output_size = x.size(2)
         output = self.wavenet_end(skip_sum, output_size)
         #output = torch.tanh(output)
         #print (f'output is {output}')
         return output
