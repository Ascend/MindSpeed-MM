# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import os
import sys
import logging

logger = logging.getLogger(__name__)


def get_checkpoint_name(checkpoints_path, iteration, release=False):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    common_path = os.path.join(checkpoints_path, directory)
    return common_path


def get_checkpoint_tracker_filename(checkpoints_path):
    """Tracker file rescords the latest chckpoint during training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def read_metadata(tracker_filename):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError as e:
            release = metastring == 'release'
            if not release:
                raise ValueError('ERROR: Invalid metadata file {}.'.format(tracker_filename)) from e
    if not (iteration > 0 or release):
        print('error parsing metadata file {}'.format(tracker_filename))

    return iteration, release