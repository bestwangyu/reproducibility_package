# Modified from songwenas12/fjsp-drl for this study.
# Original upstream attribution is retained in NOTICE and THIRD_PARTY_NOTICES.md.
# Copyright (c) 2026 Yu Wang and Xiaoyao Ding for modifications.
# SPDX-License-Identifier: Apache-2.0
from gym.envs.registration import register

# Registrar for the gym environment
# https://www.gymlibrary.ml/content/environment_creation/ for reference
register(
    id='fjsp-v0',  # Environment name (including version number)
    entry_point='env.fjsp_env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)

register(
    id='fjsp-dynamic-v0',
    entry_point='env.dynamic_fjsp_env:DynamicFJSPEnv',
)

register(
    id='fjsp-composite-dynamic-v0',
    entry_point='env.composite_dynamic_fjsp_env:CompositeDynamicFJSPEnv',
)
