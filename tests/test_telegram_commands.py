from telegram_commands import is_paused, _handle_pause, _handle_resume


def test_pause_resume_toggle():
    import telegram_commands
    assert is_paused() is False
    telegram_commands._paused = True
    assert is_paused() is True
    telegram_commands._paused = False
    assert is_paused() is False
