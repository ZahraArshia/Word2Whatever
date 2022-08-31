# Word2Whatever!
> An improvment of shallow word embeddings to challenge deep ones! deep learning is not always the solution!

![image](https://user-images.githubusercontent.com/78906545/187692286-022f40c9-bdcc-4bbd-97bf-6ef4a55885ef.png)

The need to convert human understandable text data into mathematically processable data has been the subject of researchers' studies even before the emergence of concepts such as artificial intelligence and natural language processing. In recent years, this concept has been more widely used and better-known due to the increase in the use of tasks related to natural language processing. However, since natural language processing tasks strongly depend on word embeddings, efforts are always made to provide a better and more efficient word embedding approach.

In this project, we present a method to improve the performance of word embedding. Our proposed method can address some of the problems of the current embedding methods. The provided approach is based on the existing statistical methods by changing the data structure of classical word embeddings. It can act as an improvement on previous embeddings or independently as a method for producing new embeddings.

## Quick links

- [link to the paper]()(soon)

- [link to the article]()(soon)

## Getting Started

**pre-requirements**

This project uses Python as the main programming language, however, we used Matlab for some preprocesses, so it's recommended to use the last version of python to avoid errors.

**1) Series generation**

In this phase, shallow word embeddings such as Word2Vec, GloVe or Fast text convert into series and will be treated as signals in the next step.
The "SeriesGeneration" folder contains the code related to this conversion. We used some example embeddings to generate signals however, you can use any word embedding, either shallow or deep.

**2) Building 2D structure**

In the second phase, we convert signals into the matrix structure using statistical methods. We used two different approaches to do so. You can find them in the "RP" folder. Only one of the RP files can be used on each test.

**3) Evaluation**
As explained in the paper and article, we used several evaluation methods including external and internal evaluations. In the "evaluation" folder, you can file all the methods used to do evaluations.

## Author

**Zahra Aershia**

- GitHub: [@ZahraArshia](https://github.com/ZahraArshia)
- LinkedIn: [@ZahraArshia](https://www.linkedin.com/in/ZahraArshia/)


## Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](https://github.com/ZahraArshia/Word2Whatever/issues).

## Show your support

Give a ⭐️ if you like this project!

## Acknowledgments

We would like to acknowledge [MUT NLP lab](https://github.com/mutnlp) for their support.

## License

This project is [MIT](https://github.com/ZahraArshia/Word2Whatever/blob/main/LICENSE) licensed.
