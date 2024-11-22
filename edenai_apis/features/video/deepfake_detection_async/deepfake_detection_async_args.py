HUMAN_IMAGE_EXAMPLE = ('https://d14uq1pz7dzsdq.cloudfront.net/4f8f3bb1-b322-4ef3-a166-da641d37d746_/tmp/tmp08h_rdxu.upload.mp4?Expires=1727709680&Signature=DmyYhOaki-tGoHz2QT1ekCk7B0iDBuaHQbX6kpwUM9m1WlEZqbxAN0uqDruu0g7f3i8OrUProh1Kgi91H4HciN0s74o80P8QAd7LjQmBM86aLedvAaoeGeBJis3dXQrj2jHtMyohq~yf1mRi57YbCdxcI5jxtcWUWVEdtmHi0vFGuINhWMymaFGnrKcvEXRFNdTrwpWt9Hs6mVqX9xbYuVd0UFUo1wTyFoYwL4FGbSh7tXLljsItAL7BfB5LKjMryMiD4kTe3NGOBIxPtDRfiw2FKXK0ij2ksTPeHyA3vUaB--3~l6GCIYoUS5RXyNc-GLpuCNdLb5baY8wKrumrOQ__&Key-Pair-Id=K1F55BTI9AHGIK')

def deepfake_detection_async_arguments(provider_name: str) -> dict:
    return {
        "file_url": HUMAN_IMAGE_EXAMPLE
    }
