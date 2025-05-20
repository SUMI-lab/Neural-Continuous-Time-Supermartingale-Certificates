# Neural Continuous-Time Supermartingale Certificates

This repository contains code for the paper _Neural Continuous-Time Supermartingale Certificates_ by Grigory Neustroev, Mirco Giacobbe, Anna Lukina, published in AAAI 2025. To cute this work you can use:
```
@inproceedings{neustroev2025neural,
  title={Neural continuous-time supermartingale certificates},
  author={Neustroev, Grigory and Giacobbe, Mirco and Lukina, Anna},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={26},
  pages={27538--27546},
  year={2025}
}
```

## How to Install

In the paper, all experiments we run using Python 3.11.7 and the libraries from
the [requirements file](requirements.txt).
Please, download `auto_LiRPA` manually (we used v0.5.0, specifically
[commit bfb7997](https://github.com/Verified-Intelligence/auto_LiRPA/tree/bfb7997))
and then either place in the `auto_LiRPA` subdirectory of this project, or
install it yourself.

## How to Use

Run the [GBM](run_gbm.py) and [Inverted Pendulum](run_pendulum.py) scripts. The data will be saved to .csv files (copies are included in the results directory). The columns are: random seed, runtime, verification result, final epoch number.

## Contributors

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="33.33%"><a href="https://github.com/greg-neustroev"><img src="https://avatars.githubusercontent.com/u/32451432?v=4?s=100" width="100px;" alt="Greg Neustroev"/><br /><b>Greg Neustroev</b></a></td>
      <td align="center" valign="top" width="33.33%"><a href="https://github.com/mircogiacobbe"><img src="https://avatars.githubusercontent.com/u/1612237?v=4?s=100" width="100px;" alt="Mirco Giacobbe"/><br /><b>Mirco Giacobbe</b></a></td>
      <td align="center" valign="top" width="33.33%"><a href="https://github.com/AnnaLukina"><img src="https://avatars.githubusercontent.com/u/17516017?v=4?s=100" width="100px;" alt="Anna Lukina"/><br /><b>Anna Lukina</b></a></td>
    </tr>
  </tbody>
</table>


