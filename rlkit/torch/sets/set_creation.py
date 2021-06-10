import pickle
from os import path as osp
from typing import List

from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.envs.pygame.pnp_util import sample_pnp_sets
from rlkit.misc import asset_loader
from rlkit.launchers.config import LOCAL_LOG_DIR
from rlkit.torch.sets import set


def create_sets(
        env,
        renderer,
        saved_filename=None,
        save_to_filename=None,
        example_paths_and_math_set_classes=None,
        **kwargs
) -> List[set.Set]:
    if saved_filename is not None:
        sets = asset_loader.load_local_or_remote_file(saved_filename)
    elif example_paths_and_math_set_classes is not None:
        sets = [
            create_set_object_from_examples(**kwargs)
            for kwargs in example_paths_and_math_set_classes
        ]
    else:
        if isinstance(env, PickAndPlaceEnv):
            sets = sample_pnp_sets(env, renderer, **kwargs)
        else:
            raise NotImplementedError()
    if save_to_filename:
        save(sets, save_to_filename)
    return sets


def create_set_object_from_examples(
        examples_path,
        math_set_class,
        math_set_class_kwargs=None,
):
    if math_set_class_kwargs is None:
        math_set_class_kwargs = {}
    example_dict = asset_loader.load_local_or_remote_file(examples_path).item()
    example_dict['example_image'] = example_dict['image_desired_goal']
    example_dict['example_state'] = example_dict['state_desired_goal']
    description = math_set_class(**math_set_class_kwargs)
    return set.Set(description, example_dict)


def create_debug_set(example_dict):
    debug_set = set.DebugSet()
    return set.Set(debug_set, example_dict)


def get_absolute_path(relative_path):
    return osp.join(LOCAL_LOG_DIR, relative_path)


def load(relative_path):
    path = get_absolute_path(relative_path)
    print("loading data from", path)
    return pickle.load(open(path, "rb"))


def save(data, relative_path):
    path = get_absolute_path(relative_path)
    pickle.dump(data, open(path, "wb"))