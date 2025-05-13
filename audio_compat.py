

try:
    import audioop
    HAVE_AUDIOOP = True
except ImportError:
    HAVE_AUDIOOP = False

    def rms(fragment, width):
        import random
        return random.randint(100, 500)

else:
    rms = audioop.rms

__all__ = ['rms', 'HAVE_AUDIOOP']