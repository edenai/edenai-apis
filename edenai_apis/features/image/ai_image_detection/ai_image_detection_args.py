HUMAN_IMAGE_EXAMPLE = ('https://www.iptc.org/std-dev/photometadata/examples/google-licensable/images/IPTC-GoogleImgSrcPmd_testimg01.jpg')
def ai_image_detection_arguments(provider_name: str) -> dict:
    return {
        "image_url": HUMAN_IMAGE_EXAMPLE
    }
