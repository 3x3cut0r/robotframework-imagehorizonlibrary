# -*- coding: utf-8 -*-
from unittest import TestCase
from unittest.mock import call, MagicMock, patch


class TestMouse(TestCase):

    def setUp(self):
        self.mock = MagicMock()
        self.patcher = patch.dict('sys.modules', {'pyautogui': self.mock})
        self.patcher.start()
        from ImageHorizonLibrary import ImageHorizonLibrary, MouseException
        self.lib = ImageHorizonLibrary()

    def tearDown(self):
        self.mock.reset_mock()
        self.patcher.stop()

    def test_all_directional_clicks(self):
        for direction in ['above', 'below', 'left', 'right']:
            fn = getattr(self.lib, 'click_to_the_%s_of' % direction)
            fn((0, 0), '10')
        self.assertEqual(self.mock.click.mock_calls,
                          [call(0, -10, button='left', interval=0.0, clicks=1),
                           call(0, 10, button='left', interval=0.0, clicks=1),
                           call(-10, 0, button='left', interval=0.0, clicks=1),
                           call(10, 0, button='left', interval=0.0, clicks=1)])

    def _verify_directional_clicks_fail(self, direction, kwargs):
        from ImageHorizonLibrary import MouseException

        fn = getattr(self.lib, 'click_to_the_%s_of' % direction)
        with self.assertRaises(MouseException):
            fn((0, 0), 10, **kwargs)
        self.assertEqual(self.mock.click.mock_calls, [])

    def test_arguments_in_directional_clicks(self):
        self.lib.click_to_the_above_of((0, 0), 10, clicks='2',
                                       button='middle', interval='1.2')
        self.assertEqual(self.mock.click.mock_calls, [call(0, -10,
                                                            button='middle',
                                                            interval=1.2,
                                                            clicks=2)])
        self.mock.reset_mock()
        for args in (('below', {'clicks': 'notvalid'}),
                     ('right', {'button': 'notvalid'}),
                     ('left',  {'interval': 'notvalid'})):
            self._verify_directional_clicks_fail(*args)

    def _verify_move_to_fails(self, *args):
        from ImageHorizonLibrary import MouseException
        with self.assertRaises(MouseException):
            self.lib.move_to(*args)

    def test_move_to(self):
        for args in [(1, 2), ((1, 2),), ('1', '2'), (('1', '2'),)]:
            self.lib.move_to(*args)
            self.assertEqual(self.mock.moveTo.mock_calls, [call(1, 2)])
            self.mock.reset_mock()

        for args in [(1,),
                     (1, 2, 3),
                     ('1', 'lollerskates'),
                     (('1', 'lollerskates'),)]:
            self._verify_move_to_fails(*args)

    def _verify_move_rel_fails(self, *args, **kwargs):
        from ImageHorizonLibrary import MouseException
        with self.assertRaises(MouseException):
            self.lib.move_rel(*args, **kwargs)

    def test_move_rel(self):
        for args in [(1, 2), ((1, 2),), ('1', '2'), (('1', '2'),)]:
            self.lib.move_rel(*args)
            self.assertEqual(self.mock.moveRel.mock_calls,
                             [call(1, 2, duration=0.0)])
            self.mock.reset_mock()

        self.lib.move_rel(1, 2, duration='0.5')
        self.assertEqual(self.mock.moveRel.mock_calls,
                         [call(1, 2, duration=0.5)])
        self.mock.reset_mock()

        for args in [(1,),
                     ('1', 'nope'),
                     (('1', 'nope'),)]:
            self._verify_move_rel_fails(*args)

        self._verify_move_rel_fails(1, 2, duration='not-a-number')

    def test_mouse_down(self):
        for args in [tuple(), ('right',)]:
            self.lib.mouse_down(*args)
        self.assertEqual(self.mock.mouseDown.mock_calls, [call(button='left'), call(button='right')])

    def test_mouse_up(self):
        for args in [tuple(), ('right',)]:
            self.lib.mouse_up(*args)
        self.assertEqual(self.mock.mouseUp.mock_calls, [call(button='left'), call(button='right')])

    def test_drag_to(self):
        self.lib.drag_to(10, 20)
        self.assertEqual(
            self.mock.dragTo.mock_calls,
            [call(10, 20, duration=0.0, button='left', mouseDownUp=True)]
        )
        self.mock.reset_mock()

        self.lib.drag_to((10, 20, 0.8), duration='0.5', button='middle', mouse_down_up='false')
        self.assertEqual(
            self.mock.dragTo.mock_calls,
            [call(10, 20, duration=0.5, button='middle', mouseDownUp=False)]
        )
        self.mock.reset_mock()

        from ImageHorizonLibrary import MouseException
        with self.assertRaises(MouseException):
            self.lib.drag_to(10)
        with self.assertRaises(MouseException):
            self.lib.drag_to(10, 20, duration='invalid')

    def test_drag_rel(self):
        self.lib.drag_rel(5, -5)
        self.assertEqual(
            self.mock.dragRel.mock_calls,
            [call(5, -5, duration=0.0, button='left', mouseDownUp=True)]
        )
        self.mock.reset_mock()

        self.lib.drag_rel((5, -5, 0.9), duration='1', button='right', mouse_down_up='No')
        self.assertEqual(
            self.mock.dragRel.mock_calls,
            [call(5, -5, duration=1.0, button='right', mouseDownUp=False)]
        )
        self.mock.reset_mock()

        from ImageHorizonLibrary import MouseException
        with self.assertRaises(MouseException):
            self.lib.drag_rel(5)
        with self.assertRaises(MouseException):
            self.lib.drag_rel(5, 5, duration='invalid')

    def test_scroll(self):
        self.lib.scroll('3')
        self.assertEqual(self.mock.scroll.mock_calls, [call(3)])
        self.mock.reset_mock()

        self.lib.scroll(-5, 10, 20)
        self.assertEqual(self.mock.scroll.mock_calls, [call(-5, x=10, y=20)])
        self.mock.reset_mock()

        from ImageHorizonLibrary import MouseException
        with self.assertRaises(MouseException):
            self.lib.scroll('foo')
        with self.assertRaises(MouseException):
            self.lib.scroll(1, 2)

    def test_horizontal_scroll(self):
        self.lib.scroll_horizontally('2')
        self.assertEqual(self.mock.hscroll.mock_calls, [call(2)])
        self.mock.reset_mock()

        from ImageHorizonLibrary import MouseException
        with self.assertRaises(MouseException):
            self.lib.scroll_horizontally('foo')

    def test_vertical_scroll(self):
        self.lib.scroll_vertically('-3')
        self.assertEqual(self.mock.vscroll.mock_calls, [call(-3)])
        self.mock.reset_mock()

        from ImageHorizonLibrary import MouseException
        with self.assertRaises(MouseException):
            self.lib.scroll_vertically('foo')

    def test_global_pause_and_failsafe(self):
        self.lib.set_global_pause('0.25')
        self.assertEqual(self.mock.PAUSE, 0.25)

        self.lib.set_failsafe('False')
        self.assertFalse(self.mock.FAILSAFE)

        from ImageHorizonLibrary import MouseException
        with self.assertRaises(MouseException):
            self.lib.set_global_pause('nope')
        with self.assertRaises(MouseException):
            self.lib.set_failsafe('maybe')
