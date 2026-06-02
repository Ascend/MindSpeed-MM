import os
import numpy as np
import torch
import torch.nn.functional as F


def patch_t2v():
    import vbench.subject_consistency as vbench_sub
    import vbench.background_consistency as vbench_back
    import vbench.aesthetic_quality as vbench_aes
    import vbench.imaging_quality as vbench_image
    import vbench.object_class as vbench_obj
    import vbench.multiple_objects as vbench_multi
    import vbench.color as vbench_color
    import vbench.scene as vbench_scene
    import vbench.human_action as vbench_human
    import vbench.appearance_style as vbench_appear

    vbench_sub.subject_consistency = subject_consistency
    vbench_back.background_consistency = background_consistency
    vbench_aes.laion_aesthetic = laion_aesthetic
    vbench_image.technical_quality = technical_quality
    vbench_obj.object_class = object_class
    vbench_multi.multiple_objects = multiple_objects
    vbench_color.color = color
    vbench_scene.scene = scene
    vbench_human.human_action = human_action
    vbench_appear.appearance_style = appearance_style


def subject_consistency(model, video_list, device, read_frame):
    from vbench.utils import load_video, load_dimension_info, dino_transform, dino_transform_Image
    from PIL import Image
    from tqdm import tqdm
    from vbench.distributed import get_rank

    sim = 0.0
    cnt = 0
    video_results = []
    if read_frame:
        image_transform = dino_transform_Image(224)
    else:
        image_transform = dino_transform(224)
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(image_transform(Image.open(tmp_path)))
        else:
            images = load_video(video_path)
            images = image_transform(images)
        for i, item in enumerate(images):
            with torch.no_grad():
                image = item.unsqueeze(0)
                image = image.to(device)
                image_features = model(image)
                image_features = F.normalize(image_features, dim=-1, p=2)
                if i == 0:
                    first_image_features = image_features
                else:
                    sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1
            former_image_features = image_features
        sim_per_images = video_sim / (len(images) - 1)
        sim += video_sim
        video_results.append({'video_path': video_path, 'video_results': sim_per_images})
    sim_per_frame = sim / cnt if cnt != 0 else None
    return sim_per_frame, video_results


def background_consistency(clip_model, preprocess, video_list, device, read_frame):
    from vbench.utils import load_video, clip_transform
    from PIL import Image
    from tqdm import tqdm
    from vbench.distributed import get_rank

    sim = 0.0
    cnt = 0
    video_results = []
    image_transform = clip_transform(224)
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        cnt_per_video = 0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(preprocess(Image.open(tmp_path)))
            images = torch.stack(images)
        else:
            images = load_video(video_path)
            images = image_transform(images)
        images = images.to(device)
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        for i, item in enumerate(image_features):
            image_feature = item.unsqueeze(0)
            if i == 0:
                first_image_feature = image_feature
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
                cnt_per_video += 1
            former_image_feature = image_feature
        sim_per_image = video_sim / (len(image_features) - 1)
        sim += video_sim
        video_results.append({
            'video_path': video_path,
            'video_results': sim_per_image,
            'video_sim': video_sim,
            'cnt_per_video': cnt_per_video})
    sim_per_frame = sim / cnt if cnt != 0 else None
    return sim_per_frame, video_results


def laion_aesthetic(aesthetic_model, clip_model, video_list, device):
    from vbench.utils import load_video, clip_transform
    from tqdm import tqdm
    from vbench.distributed import get_rank

    aesthetic_model.eval()
    clip_model.eval()
    aesthetic_avg = 0.0
    num = 0
    batch_size = 32
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(video_path)
        image_transform = clip_transform(224)

        aesthetic_scores_list = []
        for i in range(0, len(images), batch_size):
            image_batch = images[i:i + batch_size]
            image_batch = image_transform(image_batch)
            image_batch = image_batch.to(device)

            with torch.no_grad():
                image_feats = clip_model.encode_image(image_batch).to(torch.float32)
                image_feats = F.normalize(image_feats, dim=-1, p=2)
                aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)

            aesthetic_scores_list.append(aesthetic_scores)

        aesthetic_scores = torch.cat(aesthetic_scores_list, dim=0)
        normalized_aesthetic_scores = aesthetic_scores / 10
        cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)
        aesthetic_avg += cur_avg.item()
        num += 1
        video_results.append({'video_path': video_path, 'video_results': cur_avg.item()})

    if num != 0:
        aesthetic_avg /= num
    return aesthetic_avg, video_results


def technical_quality(model, video_list, device, **kwargs):
    from vbench.utils import load_video, clip_transform
    from tqdm import tqdm
    from vbench.distributed import get_rank
    from vbench.imaging_quality import transform

    if 'imaging_quality_preprocessing_mode' not in kwargs:
        preprocess_mode = 'longer'
    else:
        preprocess_mode = kwargs['imaging_quality_preprocessing_mode']
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(video_path)
        images = transform(images, preprocess_mode)
        acc_score_video = 0.
        for item in images:
            frame = item.unsqueeze(0).to(device)
            score = model(frame)
            acc_score_video += float(score)
        video_results.append({'video_path': video_path, 'video_results': acc_score_video / len(images)})

    average_score = sum([res['video_results'] for res in video_results]) / len(video_results) if video_results else 0

    average_score = average_score / 100.
    return average_score, video_results


def object_class(model, video_dict, device):
    from vbench.utils import load_video, clip_transform
    from tqdm import tqdm
    from vbench.distributed import get_rank
    from torchvision import transforms
    from vbench.object_class import get_dect_from_grit, check_generate

    success_frame_count, frame_count = 0, 0
    video_results = []
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        object_info = info['auxiliary_info']['object']
        for video_path in info['video_list']:
            video_tensor = load_video(video_path, num_frames=16)
            _, _, h, w = video_tensor.size()
            if min(h, w) > 768:
                scale = 720. / min(h, w)
                output_tensor = transforms.Resize(size=(int(scale * h), int(scale * w)), )(video_tensor)
                video_tensor = output_tensor
            cur_video_pred = get_dect_from_grit(model, video_tensor.permute(0, 2, 3, 1))
            cur_success_frame_count = check_generate(object_info, cur_video_pred)
            cur_success_frame_rate = cur_success_frame_count / len(cur_video_pred)
            success_frame_count += cur_success_frame_count
            frame_count += len(cur_video_pred)
            video_results.append({
                'video_path': video_path,
                'video_results': cur_success_frame_rate,
                'success_frame_count': cur_success_frame_count,
                'frame_count': len(cur_video_pred)})
    success_rate = success_frame_count / frame_count if frame_count != 0 else None
    return success_rate, video_results


def multiple_objects(model, video_dict, device):
    from vbench.utils import load_video, clip_transform
    from tqdm import tqdm
    from vbench.distributed import get_rank
    from torchvision import transforms
    from vbench.multiple_objects import get_dect_from_grit, check_generate

    success_frame_count, frame_count = 0, 0
    video_results = []
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        object_info = info['auxiliary_info']['object']
        for video_path in info['video_list']:
            video_tensor = load_video(video_path, num_frames=16)
            _, _, h, w = video_tensor.size()
            if min(h, w) > 768:
                scale = 720. / min(h, w)
                output_tensor = transforms.Resize(size=(int(scale * h), int(scale * w)), )(video_tensor)
                video_tensor = output_tensor
            cur_video_pred = get_dect_from_grit(model, video_tensor.permute(0, 2, 3, 1))
            cur_success_frame_count = check_generate(object_info, cur_video_pred)
            cur_success_frame_rate = cur_success_frame_count / len(cur_video_pred)
            success_frame_count += cur_success_frame_count
            frame_count += len(cur_video_pred)
            video_results.append({
                'video_path': video_path,
                'video_results': cur_success_frame_rate,
                'success_frame_count': cur_success_frame_count,
                'frame_count': len(cur_video_pred)})
    success_rate = success_frame_count / frame_count if frame_count != 0 else None
    return success_rate, video_results


def color(model, video_dict, device):
    from vbench.utils import load_video
    from tqdm import tqdm
    from vbench.distributed import get_rank
    from vbench.color import get_dect_from_grit, check_generate
    import cv2

    success_frame_count_all, video_count = 0, 0
    video_results = []
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        color_info = info['auxiliary_info']['color']
        object_info = info['prompt']
        object_info = object_info.replace('a ', '').replace('an ', '').replace(color_info, '').strip()
        for video_path in info['video_list']:
            video_arrays = load_video(video_path, num_frames=16, return_tensor=False)
            _, h, w, _ = video_arrays.shape
            if min(h, w) > 768:
                scale = 720.0 / min(h, w)
                new_h = int(scale * h)
                new_w = int(scale * w)
                resized_video = np.zeros((video_arrays.shape[0], new_h, new_w, 3), dtype=video_arrays.dtype)
                for i in range(video_arrays.shape[0]):
                    resized_video[i] = cv2.resize(video_arrays[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                video_arrays = resized_video
            cur_video_pred = get_dect_from_grit(model, video_arrays)
            cur_object, cur_object_color = check_generate(color_info, object_info, cur_video_pred)
            if cur_object > 0:
                cur_success_frame_rate = cur_object_color / cur_object
                success_frame_count_all += cur_success_frame_rate
                video_count += 1
                video_results.append({
                    'video_path': video_path,
                    'video_results': cur_success_frame_rate,
                    'cur_success_frame_rate': cur_success_frame_rate, })
    success_rate = success_frame_count_all / video_count if video_count != 0 else None
    return success_rate, video_results


def scene(model, video_dict, device):
    from tqdm import tqdm
    from vbench.distributed import get_rank
    from vbench.utils import load_video, load_dimension_info, tag2text_transform
    from vbench.scene import get_caption, check_generate

    success_frame_count, frame_count = 0, 0
    video_results = []
    transform = tag2text_transform(384)
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        scene_info = info['auxiliary_info']['scene']
        for video_path in info['video_list']:
            video_array = load_video(video_path, num_frames=16, return_tensor=False, width=384, height=384)
            video_tensor_list = []
            for i in video_array:
                video_tensor_list.append(transform(i).to(device).unsqueeze(0))
            video_tensor = torch.cat(video_tensor_list)
            cur_video_pred = get_caption(model, video_tensor)
            cur_success_frame_count = check_generate(scene_info, cur_video_pred)
            cur_success_frame_rate = cur_success_frame_count / len(cur_video_pred)
            success_frame_count += cur_success_frame_count
            frame_count += len(cur_video_pred)
            video_results.append({
                'video_path': video_path,
                'video_results': cur_success_frame_rate,
                'success_frame_count': cur_success_frame_count,
                'frame_count': len(cur_video_pred)})
    success_rate = success_frame_count / frame_count if frame_count != 0 else None
    return success_rate, video_results


def human_action(umt_path, video_list, device):
    from tqdm import tqdm
    from vbench.distributed import get_rank
    from vbench.utils import load_video, load_dimension_info, tag2text_transform
    from timm.models import create_model
    from vbench.third_party.umt.datasets.video_transforms import (
        Compose, Resize, CenterCrop, Normalize,
        create_random_augment, random_short_side_scale_jitter,
        random_crop, random_resized_crop_with_shift, random_resized_crop,
        horizontal_flip, random_short_side_scale_jitter, uniform_crop,
    )
    from vbench.third_party.umt.datasets.volume_transforms import ClipToTensor
    from vbench.human_action import build_dict

    state_dict = torch.load(umt_path, map_location='cpu')
    model = create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=400,
        all_frames=16,
        tubelet_size=1,
        use_learnable_pos_emb=False,
        fc_drop_rate=0.,
        drop_rate=0.,
        drop_path_rate=0.2,
        attn_drop_rate=0.,
        drop_block_rate=None,
        use_checkpoint=False,
        checkpoint_num=16,
        use_mean_pooling=True,
        init_scale=0.001,
    )
    data_transform = Compose([
        Resize(256, interpolation='bilinear'),
        CenterCrop(size=(224, 224)),
        ClipToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = model.to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    cat_dict = build_dict()
    cnt = 0
    cor_num = 0
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        cor_num_per_video = 0
        video_label_ls = video_path.split('/')[-1].lower().split('-')[0].split("person is ")[-1].split('_')[0]
        cnt += 1
        images = load_video(video_path, data_transform, num_frames=16)
        images = images.unsqueeze(0)
        images = images.to(device)
        with torch.no_grad():
            logits = torch.sigmoid(model(images))
            results, indices = torch.topk(logits, 5, dim=1)
        indices = indices.squeeze().tolist()
        results = results.squeeze().tolist()
        results = [round(f, 4) for f in results]
        cat_ls = []
        for i in range(5):
            if results[i] >= 0.85:
                cat_ls.append(cat_dict[str(indices[i])])
        flag = False
        for cat in cat_ls:
            if cat == video_label_ls:
                cor_num += 1
                cor_num_per_video += 1
                flag = True
                break
        if flag is False:
            pass
        video_results.append({
            'video_path': video_path,
            'video_results': flag,
            'cor_num_per_video': cor_num_per_video})
    acc = cor_num / cnt if cnt != 0 else None
    return acc, video_results


def appearance_style(clip_model, video_dict, device, sample="rand"):
    from tqdm import tqdm
    from PIL import Image
    import clip
    from vbench.distributed import get_rank
    from vbench.utils import load_video, clip_transform_Image

    sim = 0.0
    cnt = 0
    video_results = []
    image_transform = clip_transform_Image(224)
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        query = info['auxiliary_info']['appearance_style']
        text = clip.tokenize([query]).to(device)
        video_list = info['video_list']
        for video_path in video_list:
            cur_video = []
            with torch.no_grad():
                video_arrays = load_video(video_path, return_tensor=False)
                images = [Image.fromarray(i) for i in video_arrays]
                for image in images:
                    image = image_transform(image)
                    image = image.to(device)
                    logits_per_image, logits_per_text = clip_model(image.unsqueeze(0), text)
                    cur_sim = float(logits_per_text[0][0].cpu())
                    cur_sim = cur_sim / 100
                    cur_video.append(cur_sim)
                    sim += cur_sim
                    cnt += 1
                video_sim = np.mean(cur_video)
                video_results.append({
                    'video_path': video_path,
                    'video_results': video_sim,
                    'frame_results': cur_video,
                    'cur_sim': cur_sim})
    sim_per_frame = sim / cnt if cnt != 0 else None
    return sim_per_frame, video_results
