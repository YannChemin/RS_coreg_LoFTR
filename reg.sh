from kornia.geometry import ImageRegistrator
img_src = "HyMap_Haib_Kornia.jpg"
img_dst = "WV3_Haib_Kornia.jpg"
registrator = ImageRegistrator('similarity')
homo = registrator.register(img_src, img_dst)
plt.show()
