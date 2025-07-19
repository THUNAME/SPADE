# ServiceSeer: "Divide, Predict, Conquer: Adaptive Internet-wide Service Discovery with Limited Seeds"

## File Description

-   `spl_main.py`: main script of sampling seeds for a given dataset
-   `rec_main.py`: main script of running ServiceSeer
-   `conf.yaml`: config file of ServiceSeer

## Environment

-   Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz
-   Python version 3.9

## Getting Started

> Recommend conda environment or venv

Run the following commands in turn at bash or shell

1. `pip install -r requirements.txt`
2. `python spl_main.py -i <raw dataset dirpath> -o <saving path of sampled seeds> -c <config filepath> -r <maximum sampling ratio>`
3. `python rec_main.py -i <raw dataset dirpath> -o <saving dirpath of results> -c <config filepath> -s <sampled seeds filepath>`

All result will be saved in `<saving dirpath of results>`

Examples:

1. `python spl_main.py -i "./datasets/ipv4" -o "./ipv4.csv" -c "conf.yaml" -r 0.01`
2. `python rec_main.py -i "./datasets/ipv4" -o "./results/ipv4" -c "conf.yaml" -s "./ipv4.csv"`

## Datasets of ServiceSeer

### Offline Datasets

We conducted offline experiments on four offline datasets:
| Dataset | Source | Date | #Hosts | #(port, service) | #BGP | #AS | #IC |
| :-----: | :-------------------------------: | :----------------------------: | :-----------: | :--------------: | :---: | :---: | :---: |
| D1 | Censys | December 2020 | 101M | 190M | 660K | 67K | 79 |
| D2 | Censys | December 2020 | 31M | 54M | 539K | 60K | 79 |
| D3 | IPv6 Hitlist & low-speed scanning | December 2024 to February 2025 | 0.96M | 2.72M | 26K | 11K | 70 |
| D4 | IPv6 Hitlist & low-speed scanning | December 2024 to February 2025 | 0.74M | 1.97M | 22K | 10K | 70 |

### Data Open Source Notice

#### Usage Restrictions

This data is only allowed to be used for academic research. Any form of commercial use, data resale, or other non-academic use is strictly prohibited. Without permission, the data shall not be used to develop commercial products, conduct profit-making analysis, or disseminate it to commercial institutions.

#### Acquisition Method

If you need to obtain the data, please send an email to [blindreviewanon@163.com] using your academic institution email. The email subject should indicate: ServiceSeer [Dataset] Application for Academic Use - [Applicant's Name] - [Affiliated Institution]. The content of the email should include the following information:

-   Applicant's name, affiliated academic institution, and title/identity (such as graduate student, researcher, etc.).
-   Specific research project name, research purpose, and brief content for which the data is planned to be used.
-   The required data scale, including the quantity, scope and specific types of data needed.
-   A commitment to using the data solely for academic research and not for commercial use or illegal dissemination.

#### Review Process

We will review the email within 7 working days after receiving it. After the review is passed, we will send you the data acquisition link and usage authorization instructions via email. If the application is not approved, the specific reason will also be informed.

#### Liability Statement

Since these data are sensitive to some of the probed networks, if it is found that the data user violates the agreement of academic use, we have the right to terminate the data usage authorization immediately and reserve the right to pursue legal liability. The data user shall bear all relevant responsibilities arising from the use of the data, and our side shall not be responsible for any problems that may occur during the data usage process.

We are committed to promoting academic cooperation and knowledge progress. Thank you for your understanding and cooperation! If you have any questions, please feel free to contact us at [blindreviewanon@163.com].
