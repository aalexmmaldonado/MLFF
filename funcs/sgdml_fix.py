"""Fixes issue with TypeError: test() got multiple values for argument 'n_test'"""

import os
import numpy as np
import sys
import traceback

from sgdml.utils import ui, io
from sgdml.predict import GDMLPredict
from sgdml.train import GDMLTrain

import logging
log = logging.getLogger('sgdml')

def train(
    task_dir, valid_dataset, overwrite, max_processes, use_torch, command=None, **kwargs
):

    task_dir, task_file_names = task_dir
    n_tasks = len(task_file_names)

    func_called_directly = (
        command == 'train'
    )  # has this function been called from command line or from 'all'?
    if func_called_directly:
        ui.print_step_title('MODEL TRAINING')

    def cprsn_callback(n_atoms, n_atoms_kept):
        log.info(
            '{:d} out of {:d} atoms remain after compression.\n'.format(
                n_atoms_kept, n_atoms
            )
            + 'Note: Compression reduces the size of the optimization problem the cost of prediction accuracy!'
        )

    unconv_model_file = '_unconv_model.npz'
    unconv_model_path = os.path.join(task_dir, unconv_model_file)

    def save_progr_callback(
        unconv_model,
    ):  # saves current (unconverged) model during iterative training

        np.savez_compressed(unconv_model_path, **unconv_model)

    try:
        gdml_train = GDMLTrain(max_processes=max_processes, use_torch=use_torch)
    except:
        print()
        log.critical(traceback.format_exc())
        sys.exit()

    prev_valid_err = -1

    for i, task_file_name in enumerate(task_file_names):
        if n_tasks > 1:
            if i > 0:
                print()
            print(ui.white_bold_str('Task {:d} of {:d}'.format(i + 1, n_tasks)))

        task_file_path = os.path.join(task_dir, task_file_name)
        with np.load(task_file_path, allow_pickle=True) as task:

            model_file_name = io.model_file_name(task, is_extended=False)
            model_file_path = os.path.join(task_dir, model_file_name)

            if not overwrite and os.path.isfile(model_file_path):
                log.warning(
                    'Skipping exising model \'{}\'.'.format(model_file_name)
                    + (
                        '\nRun \'{} train -o {}\' to overwrite.'.format(
                            PACKAGE_NAME, task_file_path
                        )
                        if func_called_directly
                        else ''
                    )
                )
                continue

            try:
                model = gdml_train.train(
                    task, cprsn_callback, save_progr_callback, ui.callback
                )
            except:
                print()
                log.critical(traceback.format_exc())
                sys.exit()
            else:
                if func_called_directly:
                    log.done('Writing model to file \'{}\''.format(model_file_path))
                np.savez_compressed(model_file_path, **model)

                # Delete temporary model, if one exists.
                unconv_model_exists = os.path.isfile(unconv_model_path)
                if unconv_model_exists:
                    os.remove(unconv_model_path)

                # Delete template model, if one exists.
                templ_model_path = os.path.join(task_dir, 'm0.npz')
                templ_model_exists = os.path.isfile(templ_model_path)
                if templ_model_exists:
                    os.remove(templ_model_path)

            # Validate model.
            model_dir = (task_dir, [model_file_name])

            ###   CHANGED   ###
            # MLFF does this itself, so we shouldn't do this here
            """
            # Remove n_test if in kwargs
            if 'n_test' in kwargs.keys():
                del kwargs['n_test']
            valid_errs = test(
                model_dir,
                valid_dataset,
                -1,  # n_test = -1 -> validation mode
                overwrite,
                max_processes,
                use_torch,
                command,
                **kwargs
            )

            if prev_valid_err != -1 and prev_valid_err < valid_errs[0]:
                print()
                log.warning(
                    'Skipping remaining training tasks, as validation error is rising again.'
                )
                break

            prev_valid_err = valid_errs[0]
            """
            ###   UNCHANGED   ###

    model_dir_or_file_path = model_file_path if n_tasks == 1 else task_dir
    if func_called_directly:

        model_dir_arg = io.is_dir_with_file_type(model_dir_or_file_path, 'model', or_file=True)
        model_dir, model_files = model_dir_arg
        _print_next_step('train', model_dir=model_dir, model_files=model_files)

    return model_dir_or_file_path  # model directory or file