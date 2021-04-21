# audio8
API examples using [8-mile](https://github.com/dpressel/mead-baseline) for audio

## Implementation

The codebase relies primarily on `8-mile` (`mead-layers`) for its modeling and optimization code.
Whats left is pretty much just training and inference code

## Dependencies

The code depends on:
  - `editdistance` (for error evaluation)
  - `numpy`
  - `six`
  - `soundfile`
  - `mead-baseline`
  - `pytorch`

There are a few optional dependencies
  - `scipy` (for on-the-fly resampling of wav files)
  - `ctcdecode` (for prefix beam decoding with optional LM)
