"""Exceptions shared by the patch controller and its phases."""


class AutopatchError(Exception):
    def __init__(self, message = 'Automatic patching error'):
        self.message = message

    def __str__(self):
        return self.message
