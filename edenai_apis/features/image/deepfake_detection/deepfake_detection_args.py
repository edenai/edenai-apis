dnld_t_dpf = ("https://cdn.discordapp.com/attachments/1285184350261870694/1288046846865969163/donald_trump_Deepfake1.jpeg?ex=66f3c29b&is=66f2711b&hm=adda1bc1ae8253fd7722745d4ecdc827873cfa819e8a1abbea65e1cf80545703&")
t_cruise_not_dpf = ("https://cdn.discordapp.com/attachments/1285184350261870694/1288046848560463892/tom-cruise_Deepfake2.jpeg?ex=66f6659b&is=66f5141b&hm=0eeade51129487243fddf2424f17e5cf24f640a4ecade235d87c1dab8c7e442b&")


HUMAN_IMAGE_EXAMPLE = t_cruise_not_dpf

def deepfake_detection_arguments(provider_name: str) -> dict:
    return {
        "file_url": HUMAN_IMAGE_EXAMPLE
    }
