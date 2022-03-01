---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Ethical Blind-Spots and Possible Solutions within Medical Data Handling"
subtitle: "The Promise of Federated Machine Learning"
summary: "An Essay on how federated machine learning might solve the ethical access to Data in Healthcare."
authors: 
  - admin
tags: 
 - opinion
categories: []
date: 2021-10-05T14:04:27+01:00
lastmod: 2021-10-05T14:04:27+01:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

## Ethical Blind-Spots and Possible Solutions within Medical Data Handling

This essay is partially inspired by the Article published on Nature’s website in 2019 titled „Google Health-Data Scandal spooks researchers” written by Heidi Ledford. (Ledford, 2019). The topic is access to medical data, explored through Google’s Nightingale project in which it has partnered with Ascension one of the largest medical providers within the United States. In this partnership Google has access to the patients’ medical data including identifiable information. The article highlights the possible repercussions from starting this partnership without asking consent from the patients or informing them about it, which erodes trust in health studies in general and might hinder academic research, by limiting future data sharing.
The questions around medical data access are interesting in the context of ethical blind spots, because the promise of saving lives and improving patient experience in hospitals is such a positive promise that ethical missteps in the process can be easily overlooked or justified. While the Nightingale project might comply with regulation regarding the usage of data it doesn’t mean it is necessarily ethical, as the sharing of sensitive patient information without knowledge or consent from the patient is a violation of patients trust and right to privacy. The ethical step while at the time, not required by law would have been to give the option to opt out from the program to Ascension’s patients or decide which part of the data they are willing to share. The other ethical requirement in this case would have been to inform the patients which has not happened, while both parties maintain they complied with regulations the fact that this has prompted an investigation by US lawmakers underlines the fact companies were operating not just in a morally, but a regulatorily grey area as well.
The companies involved likely knew that the Nightingale project was ethically questionable as they have started it in secret without a press statement. Google Maintains they were not secretive as they mentioned it within their Q2 earnings call the year they started it, however the target audience of those calls are limited and they had a very limited if any chance of successfully informing the patients whose data is being collected (Shauka, 2019). The situation also likely involved delegation of responsibility on multiple fronts. Ascension as the data provider should have been the one to inform the patients and Google likely felt that this is their responsibility. The fact that Google had access to sensitive information is partially Ascension’s fault and partially Google’s. The responsibility here falls with both companies as they conflated two projects with different scopes into one according to the Google blogpost cited earlier. The project involved migrating the data infrastructure of Ascension to the Google cloud and at the same setting up data collection for machine learning development. These should have been done in separate projects, first migrate the data centres to the cloud, then Ascension should have checked and scrapped sensitive information from the data that it handed over to Google for machine learning. The current setup of the project creates an inner conflict of interest within Google, as it is tasked with safeguarding and limiting access to the same dataset that it can use to create valuable machine learning tools. DeepMind a company owned by Google handled this in a better way in the UK when they partnered with the NHS in 2016 the provided data was stored at a third party contracted by Google for the duration of the project, which reduces the conflict of interest mentioned in the previous case. However even that deal is now part of a class action lawsuit as CNBC reports (Shead, 2021). What these situations highlight is that when it comes to data, regulation often plays catch up with new technologies. Also, just because a company is in line with the regulations of the time doesn’t mean their actions are ethical. These situations have likely formed due to the incentive structure within the companies focused on results first approach, blinded by the benevolent mission of saving lives and treating data as a commodity rather than information connected to individual lives.

When it comes to questionable data decisions with such sensitive information as one’s medical records, eroding the public’s trust is very costly, it can hinder the further development of life saving technologies and strengthen the monopolies of large corporations over the data. If public trust deteriorates to a point where no medical data is shared anymore, the companies that had or still have access to it will be at an unfair competitive advantage when it comes to developing medical machine learning tools. This would also slow academic research and would reduce the amount of knowledge available within the public domain.
To avoid this situation better regulation is required, but also better systems could be designed when it comes to data handling. Federated Machine learning shows promise in this regard as it enables machine learning without data aggregation and there are signs that this method has piqued the interest of the medical community (Rieke et al., 2020). Solving the Nightingale project using federated machine learning would have meant, that the patient data never leaves the servers from the facility instead the model would have run in each of Ascensions 2600 locations and the models would be combined to a produce a better larger model in a central data centre. Federated learning gives much better privacy control to individuals, and it also levels the playing field between large companies, academics, and start-ups, allowing competition to be built around who produces the best model rather then who can acquire the most data.
While federated learning is not perfect as models can be used to extract subsets of the training data (Song, Ristenpart and Shmatikov, 2017), such breaches however still carry lower risk then a breach to a centralised database. To ensure the privacy of federated learning it needs to be paired with secure aggregation which encrypts the models sent back to the server until they have been merged into a larger model. Or differential privacy can be used, which adds noise to the data, however not all models can be adapted well for differential privacy.

For the mentioned use case of medical data handling, I believe that federated learning combined with secure aggregation would be the most functional solution while maintaining data privacy. This would require additional investment from healthcare providers compared to sharing data with a single solutions providers, but it would promote competition in the model development which could lead to faster breakthroughs and improvements for patients.

## Bibliography

Ledford, H. (2019) ‘Google health-data scandal spooks researchers’, Nature. doi: 10.1038/d41586-019-03574-5.

Rieke, N. et al. (2020) ‘The future of digital health with federated learning’, npj Digital Medicine, 3(1), p. 119. doi: 10.1038/s41746-020-00323-1.

Shauka, T. (2019) Our partnership with Ascension, Google Cloud Blog. Available at: https://cloud.google.com/blog/topics/inside-google-cloud/our-partnership-with-ascension (Accessed: 13 October 2021).

Shead, S. (2021) ‘Google and DeepMind face lawsuit over deal with Britain’s National Health Service’, CNBC, 11 October. Available at: https://www.cnbc.com/2021/10/01/google-deepmind-face-lawsuit-over-data-deal-with- britains-nhs.html.

Song, C., Ristenpart, T. and Shmatikov, V. (2017) ‘Machine learning models that remember too much’, Proceedings of the ACM Conference on Computer and Communications Security, pp. 587–601. doi: 10.1145/3133956.3134077.
