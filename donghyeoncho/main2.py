from optimization.video_editor import VideoEditor
# from optimization.video_editor_no_mask import VideoEditor
# from optimization2.image_editor2 import VideoEditor
# from optimization.copy_editor import copyEditor
from optimization.arguments import get_arguments
import natsort
import os


if __name__ == "__main__":
    args = get_arguments()
    image_editor = VideoEditor(args)
    image_editor.edit_video_by_prompt()

    # args = get_arguments()
    # image_list = natsort.natsorted(os.listdir(args.init_image))
    # mask_list = natsort.natsorted(os.listdir(args.mask))
    # assert len(image_list) == len(mask_list)
    # image_path = args.init_image
    # mask_path = args.mask
    # for i in range(len(image_list)):
    #     args.add_num = i*2
    #     args.init_image = os.path.join(image_path, image_list[i])
    #     args.mask = os.path.join(mask_path, mask_list[i])
    #     video_editor = VideoEditor(args)
    #     video_editor.edit_video_by_prompt()
    

    # image_editor.reconstruct_image()