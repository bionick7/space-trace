import numpy as np
import os.path
import pyray as rl
from typing import (
    Literal
)

DEFAULT_WINDOWN_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_FRAME_UUID = -1


# COLOR HANDLING
# ==============

ColorType = tuple[float, float, float] | str

DEFAULT_WINDOWN_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_FRAME_UUID = -1


# COLOR HANDLING
# ==============

class Color():
    '''
    Simple class to handle colors.
    '''
    def __init__(self, r: float, g: float, b: float):
        self.rgb = (r, g, b)
        self.array = np.array([*self.rgb, 1], np.float32)
        self.rl_color = rl.Color(int(r*255), int(g*255), int(b*255), 255)

    def as_rl_color(self) -> rl.Color:
        return self.rl_color
    
    def as_array(self) -> rl.Color:
        return self.array
    
    def __getstate__(self) -> dict:
        return {
            'rgb': self.rgb,
        }

    def __setstate__(self, state: dict):
        self.rgb = state['rgb']
        r, g, b = self.rgb
        self.array = np.array([*self.rgb, 1], np.float32)
        self.rl_color = rl.Color(int(r*255), int(g*255), int(b*255), 255)


def hex_to_color(hex: int) -> Color:
    return Color(
        ((hex >> 16) & 0xFF) / 255,
        ((hex >> 8) & 0xFF) / 255, 
        (hex & 0xFF) / 255
    )


class Presets:
    Earth = {
        'color': 'blue',
        'radius': 6.731e6,
        'shape': 'sphere',
        'name': 'Earth',
        'albedo_map_path': 'earthmap1k.jpg',
    }

    Moon = {
        'color': 'grey',
        'radius': 1.738e6,
        'shape': 'sphere',
        'name': 'Moon',
        'albedo_map_path': 'moonmap1k.jpg',
    }

    def _with(preset: dict, **kwargs) -> dict:
        return {**preset, **kwargs}
    
    @staticmethod
    def Earth_with(**kwargs) -> dict: return Presets._with(Presets.Earth, **kwargs)
    @staticmethod
    def Moon_with(**kwargs) -> dict: return Presets._with(Presets.Moon, **kwargs)


class Themes:
    @staticmethod
    def default_palette(name: str) -> Color:
        '''
        Default color palette for spacetrace.
        Simple function that returns the corresponding RGB values for a given color name.
        Returns aggressive magenta as error color.

        Pallette is a modification of https://lospec.com/palette-list/offshore
        '''
        
        match name:
            case 'bg':    
                return hex_to_color(0x12141c)
            case 'blue':  
                return hex_to_color(0x454e7e)
            case 'green': 
                return hex_to_color(0x4Fc76C)
            case 'red':   
                return hex_to_color(0xFF5155)
            case 'white': 
                return hex_to_color(0xfaf7d5)
            case 'gray':  
                return hex_to_color(0x735e4c)
            case 'main':
                return Themes.default_palette('white')
            case 'accent':
                return Themes.default_palette('blue')
            case 'grey':
                return Themes.default_palette('gray')
            
            case 'x-axis': return Themes.default_palette('red')
            case 'y-axis': return Themes.default_palette('green')
            case 'z-axis': return Themes.default_palette('blue')
            
        return hex_to_color(0xFF0000)

    @staticmethod
    def berry_nebula(name: str) -> Color:
        match name:

            case 'bg':     return hex_to_color(0x0d001a)
            case 'blue':   return hex_to_color(0x6d85a5)
            case 'white':  return hex_to_color(0x6cb9c9)
            case 'gray':   return hex_to_color(0x6e5181)
            case 'red':    return hex_to_color(0x6f1d5c)
            case 'main':   return Themes.berry_nebula('white')
            case 'accent': return Themes.berry_nebula('red')
            case 'grey':   return Themes.berry_nebula('gray')
            case 'x-axis': return Themes.berry_nebula('red')
            case 'y-axis': return Themes.berry_nebula('green')
            case 'z-axis': return Themes.berry_nebula('blue')
        
        # Acts as a fallback
        return Themes.default_palette(name)

def get_local_or_global_path(local_dir: str, filepath: str):
    ''' returns <local_dir>/<filepath> if it exists, otherwise returns filepath '''
    local_path = os.path.join(local_dir, filepath)
    if os.path.exists(local_path):
        return local_path
    return filepath

def transform_vectors_to_draw_space(inp: np.ndarray) -> np.ndarray:
    return inp[:,(0,2,1)] * np.array([1,1,-1])[np.newaxis,:]