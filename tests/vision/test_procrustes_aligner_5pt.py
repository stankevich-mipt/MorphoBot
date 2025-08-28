#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""Unit tests for the vision.procrustes_aligner_5pt module.

Covers:
    - 5-point landmark extraction
    - trivial and exactly solvable Procrustes benchmarks
    - .json template loading
    - forward/reverse warping with FivePointAligner class instance
"""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from vision.procrustes_aligner_5pt import (
    FivePointAligner,
    invert_similarity,
    load_template,
    select_five_from_68,
    similarity_procrustes,
    Template,
    warp_back_from_template,
    warp_to_template,
    WarpContext,
)

@pytest.fixture
def sample_68_landmarks():
    """Generate synthetic test set of 68pt landmarks."""
    landmarks = np.random.uniform(50, 200, (68, 2))
    # left eye
    landmarks[36:42] = [
        [60, 80], [65, 75], [70, 75],
        [75, 80], [70, 85], [65, 85]
    ]
    # right eye
    landmarks[42:48] = [
        [90, 80], [95, 75], [100, 75],
        [105, 80], [100, 85], [95, 85]
    ]
    # nose tip
    landmarks[30] = [82, 95]
    # left mouth corner
    landmarks[48] = [70, 110]
    # right mouth corner
    landmarks[54] = [95, 110]

    return landmarks


@pytest.fixture
def sample_image():
    """Generate  test BGR image."""
    return np.random.randint(
        0, 255, (200, 200, 3), dtype=np.uint8
    )


@pytest.fixture
def sample_templates(tmp_path_factory):
    """Create sample male and female template files."""
    tmp_path = tmp_path_factory.mktemp('templates')

    male_template_data = {
        "points": [
            [60, 80], [100, 80],
            [80, 95], [70, 100], [90, 110]
        ],
        "size": 150
    }
    female_template_data = {
        "points": [
            [65, 85], [95, 85],
            [80, 100], [72, 115], [88, 115]
        ],
        "size": 150
    }

    male_path = tmp_path / "male_template.json"
    female_path = tmp_path / "female_template.json"

    male_path.write_text(json.dumps(male_template_data))
    female_path.write_text(json.dumps(female_template_data))

    return str(male_path), str(female_path)


def test_select_five_from_68_returns_correct_shape(sample_68_landmarks):
    """Picked points form [5, 2] numpy array."""
    result = select_five_from_68(sample_68_landmarks)
    assert result.shape == (5, 2)
    assert result.dtype == np.float32


def test_select_five_from_68_computes_eye_centers(sample_68_landmarks):
    """Eye centers are computed correctly."""
    result = select_five_from_68(sample_68_landmarks)
    expected_left_eye = sample_68_landmarks[36:42].mean(0)
    np.testing.assert_array_almost_equal(
        result[0], expected_left_eye, decimal=5
    )
    expected_right_eye = sample_68_landmarks[42:48].mean(0)
    np.testing.assert_array_almost_equal(
        result[1], expected_right_eye, decimal=5
    )


def test_similarity_procrustes_identity_transform():
    """Warp between the same points is eye + zero shift."""
    src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    M, err = similarity_procrustes(src, src)

    assert M.shape == (2, 3)
    assert err < 1e-6
    np.testing.assert_array_almost_equal(
        M[:, :2], np.eye(2), decimal=5
    )
    np.testing.assert_array_almost_equal(
        M[:, 2], [0, 0], decimal=5
    )


def test_similarity_procrustes_pure_translation():
    """Warp for pure translation is eye + specified shift."""
    src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    shift = [10, 5]
    dst = src + shift
    M, err = similarity_procrustes(src, dst)
    assert err < 1e-6
    np.testing.assert_array_almost_equal(
        M[:, :2], np.eye(2), decimal=5
    )
    np.testing.assert_array_almost_equal(
        M[:, 2], shift, decimal=5
    )


def test_invert_similarity_creates_inverse():
    """Invert_similarity produces correct inverse to affine."""
    M = np.array([[2, 0, 10], [0, 2, 5]], dtype=np.float32)
    M_inv = invert_similarity(M)

    src = np.array([1, 1, 1])
    transformed = M @ src
    recovered = M_inv @ np.append(transformed, [1])

    np.testing.assert_array_almost_equal(recovered, src[:2], decimal=5)


def test_warp_to_template_valid_transform(sample_image):
    """Warp to template output signature matches the declared one."""
    src_pts = np.array(
        [
            [50, 50], [150, 50],
            [100, 75], [75, 125], [125, 125]
        ], dtype=np.float32
    )
    dst_pts = np.array(
        [
            [60, 60], [140, 60],
            [100, 85], [80, 130], [120, 130]
        ], dtype=np.float32
    )
    aligned, M, err = warp_to_template(
        sample_image, src_pts, dst_pts, (200, 200)
    )
    assert aligned is not None
    assert M is not None
    assert aligned.shape == (200, 200, 3)
    assert M.shape == (2, 3)
    assert err >= 0


def test_warp_back_from_template_shape(sample_image):
    """Function warp_back_from_template returns correct shape."""
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    result = warp_back_from_template(sample_image, M, (150, 150))

    assert result.shape == (150, 150, 3)


def test_load_template_creates_correct_template(sample_templates):
    """Loaded templates have required signature."""
    male_path, _ = sample_templates
    template = load_template(male_path)

    assert isinstance(template, Template)
    assert template.points.shape == (5, 2)
    assert template.target_size == 150
    assert template.points.dtype == np.float32


def test_five_point_aligner_initialization(sample_templates):
    """Class FivePointAligner properly instantiates from json templates."""
    male_path, female_path = sample_templates
    male_template = load_template(male_path)
    female_template = load_template(female_path)

    aligner = FivePointAligner(male_template, female_template)
    assert aligner.male == male_template
    assert aligner.female == female_template
    assert aligner.target_size == 150


def test_five_point_aligner_mismatched_sizes_raises_error(sample_templates):
    """Mismatched template sizes for FivePointAligner raise error."""
    male_path, female_path = sample_templates
    male_template = load_template(male_path)
    female_template = load_template(female_path)

    female_template = Template(female_template.points, 200)

    with pytest.raises(
        ValueError, match="All templates must share the target size"
    ):
        FivePointAligner(male_template, female_template)


def test_dst_for_label_returns_correct_template(sample_templates):
    """Dst to label mapping is appropriate."""
    male_path, female_path = sample_templates
    male_template = load_template(male_path)
    female_template = load_template(female_path)

    aligner = FivePointAligner(male_template, female_template)

    assert aligner.dst_for_label("male") == male_template
    assert aligner.dst_for_label("female") == female_template
    assert aligner.dst_for_label("unknown") == male_template  # default


def test_five_point_aligner_align_produces_valid_outputs(
    sample_templates, sample_image, sample_68_landmarks
):
    """Output signature for FivePointAligner.align matches declaration."""
    male_path, female_path = sample_templates
    male_template = load_template(male_path)
    female_template = load_template(female_path)

    aligner = FivePointAligner(male_template, female_template)
    aligned, ctx, err = aligner.align(sample_image, sample_68_landmarks, "male")

    assert aligned is not None
    assert ctx is not None
    assert aligned.shape == (150, 150, 3)
    assert isinstance(ctx, WarpContext)
    assert ctx.M.shape == (2, 3)
    assert err >= 0


def test_five_point_aligner_reverse_warp_returns_original_shape(
    sample_templates, sample_image, sample_68_landmarks
):
    """Forward->backward warp sequence retains source image shape."""
    male_path, female_path = sample_templates
    male_template = load_template(male_path)
    female_template = load_template(female_path)

    aligner = FivePointAligner(male_template, female_template)
    aligned, ctx, _ = aligner.align(sample_image, sample_68_landmarks)

    if aligned is not None and ctx is not None:
        back = aligner.reverse_warp(aligned, ctx)
        assert back.shape == sample_image.shape
