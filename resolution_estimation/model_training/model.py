class UNet3D(nn.Module):
    def __init__(
        self,
        dropout_rates=DROPOUT_RATES,
        upsample_mode="nearest",
        regularization="dropout",
        align_corners=False,
    ) -> None:
        super().__init__()

        self.reg = regularization
        self.align_corners = align_corners
        assert self.reg in {"dropout", "batchnorm"}, "Unknown regularization parameter"

        # First enc
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1, out_channels=LAYER_FILTERS[0], kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[0],
                out_channels=LAYER_FILTERS[0],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=LAYER_FILTERS[0])
            if self.reg == "batchnorm"
            else nn.Dropout3d(p=dropout_rates[0]),
        )
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)

        # Second enc
        self.conv2_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[0],
                out_channels=LAYER_FILTERS[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[1],
                out_channels=LAYER_FILTERS[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=LAYER_FILTERS[1])
            if self.reg == "batchnorm"
            else nn.Dropout3d(p=dropout_rates[1]),
        )
        self.max_pool2 = nn.MaxPool3d(kernel_size=2)

        # Third enc
        self.conv3_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[1],
                out_channels=LAYER_FILTERS[2],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[2],
                out_channels=LAYER_FILTERS[2],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=LAYER_FILTERS[2])
            if self.reg == "batchnorm"
            else nn.Dropout3d(p=dropout_rates[2]),
        )
        self.max_pool3 = nn.MaxPool3d(kernel_size=2)

        # Fourth enc
        self.conv4_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[2],
                out_channels=LAYER_FILTERS[3],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[3],
                out_channels=LAYER_FILTERS[3],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=LAYER_FILTERS[3])
            if self.reg == "batchnorm"
            else nn.Dropout3d(p=dropout_rates[3]),
        )
        self.max_pool4 = nn.MaxPool3d(kernel_size=2)

        # Root enc
        self.conv_root1 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[3],
                out_channels=LAYER_FILTERS[4],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.conv_root2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[4],
                out_channels=LAYER_FILTERS[4],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=LAYER_FILTERS[4])
            if self.reg == "batchnorm"
            else nn.Dropout3d(p=dropout_rates[4]),
        )

        ########################################################################
        # Fourth dec
        self.upsample4 = nn.Upsample(
            scale_factor=2, mode=upsample_mode, align_corners=self.align_corners
        )
        self.deconv4_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[4],
                out_channels=LAYER_FILTERS[3],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.deconv4_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[4],
                out_channels=LAYER_FILTERS[3],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=LAYER_FILTERS[3],
                out_channels=LAYER_FILTERS[3],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        # Third dec
        self.upsample3 = nn.Upsample(
            scale_factor=2, mode=upsample_mode, align_corners=self.align_corners
        )
        self.deconv3_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[3],
                out_channels=LAYER_FILTERS[2],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.deconv3_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[3],
                out_channels=LAYER_FILTERS[2],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=LAYER_FILTERS[2],
                out_channels=LAYER_FILTERS[2],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        # Second dec
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode=upsample_mode, align_corners=self.align_corners
        )
        self.deconv2_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[2],
                out_channels=LAYER_FILTERS[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.deconv2_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[2],
                out_channels=LAYER_FILTERS[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=LAYER_FILTERS[1],
                out_channels=LAYER_FILTERS[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        # First dec
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode=upsample_mode, align_corners=self.align_corners
        )
        self.deconv1_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[1],
                out_channels=LAYER_FILTERS[0],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.deconv1_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=LAYER_FILTERS[1],
                out_channels=LAYER_FILTERS[0],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=LAYER_FILTERS[0],
                out_channels=LAYER_FILTERS[0],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=LAYER_FILTERS[0], out_channels=1, kernel_size=3, padding=1
            ),
        )

    def forward(self, x):
        first_enc_res = self.conv1_2(self.conv1_1(x))
        second_enc_res = self.conv2_2(self.conv2_1(self.max_pool1(first_enc_res)))
        third_enc_res = self.conv3_2(self.conv3_1(self.max_pool2(second_enc_res)))
        fouth_enc_res = self.conv4_2(self.conv4_1(self.max_pool3(third_enc_res)))
        root_enc_res = self.conv_root2(self.conv_root1(self.max_pool4(fouth_enc_res)))

        fourth_dec_res = self.deconv4_1(self.upsample4(root_enc_res))
        fourth_dec_res = torch.cat([fouth_enc_res, fourth_dec_res], dim=1)
        fourth_dec_res = self.deconv4_2(fourth_dec_res)

        third_dec_res = self.deconv3_1(self.upsample3(fourth_dec_res))
        third_dec_res = torch.cat([third_enc_res, third_dec_res], dim=1)
        third_dec_res = self.deconv3_2(third_dec_res)

        second_dec_res = self.deconv2_1(self.upsample3(third_dec_res))
        second_dec_res = torch.cat([second_enc_res, second_dec_res], dim=1)
        second_dec_res = self.deconv2_2(second_dec_res)

        first_dec_res = self.deconv1_1(self.upsample3(second_dec_res))
        first_dec_res = torch.cat([first_enc_res, first_dec_res], dim=1)
        first_dec_res = self.deconv1_2(first_dec_res)

        return first_dec_res
