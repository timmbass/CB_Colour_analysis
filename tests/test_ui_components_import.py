from ui.components import plot_season_map, plot_season_scores


def test_ui_components_importable():
    assert callable(plot_season_map)
    assert callable(plot_season_scores)
