"""身体渲染器测试：姿态/关节/交互点/朝向镜像。

Tests for the pixel body (original blueprint sprite, articulated)."""

from __future__ import annotations

import unittest

from newworld.body import BODY_H, BODY_W, BodyRenderer, Pose


class TestBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.renderer = BodyRenderer()

    def test_original_sprite_dimensions(self) -> None:
        frame = self.renderer.render(Pose("idle", 0.0, 1))
        self.assertEqual(frame.image.size, (BODY_W, BODY_H))
        self.assertEqual((BODY_W, BODY_H), (23, 29))
        self.assertEqual(frame.anchor, (11, 28))

    def test_anatomy_points(self) -> None:
        frame = self.renderer.render(Pose("idle", 0.0, 1))
        self.assertEqual(len(frame.hands), 2)
        self.assertEqual(len(frame.feet), 2)
        for x, y in frame.hands + frame.feet + (frame.head_center, frame.mouth):
            self.assertTrue(0 <= x < BODY_W)
            self.assertTrue(0 <= y < BODY_H)

    def test_walk_animates_legs(self) -> None:
        idle = self.renderer.render(Pose("idle", 0.0, 1)).image.tobytes()
        walk = self.renderer.render(Pose("walk", 1.2, 1)).image.tobytes()
        jump = self.renderer.render(Pose("jump", 0.0, 1)).image.tobytes()
        self.assertNotEqual(idle, walk)
        self.assertNotEqual(idle, jump)

    def test_facing_mirrors(self) -> None:
        right = self.renderer.render(Pose("idle", 0.0, 1))
        left = self.renderer.render(Pose("idle", 0.0, -1))
        self.assertEqual(right.anchor[0] + left.anchor[0], BODY_W - 1)
        self.assertNotEqual(right.image.tobytes(), left.image.tobytes())

    def test_sleep_is_lying(self) -> None:
        frame = self.renderer.render(Pose("sleep", 0.0, 1))
        self.assertEqual(frame.image.size, (BODY_H, BODY_W))
        self.assertEqual(frame.hands, ())


if __name__ == "__main__":
    unittest.main()
