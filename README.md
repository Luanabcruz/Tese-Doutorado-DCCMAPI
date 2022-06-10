# Tese de Doutorado - DCCMAPI

Tese de autoria de Luana Batista da Cruz, sob o título **“Segmentação automática de rins e tumores renais em imagens de tomografia computadorizada baseada em aprendizado profundo 2.5D”**, apresentada ao Curso de Doutorado em Ciência da Computação - Associação UFMA/UFPI, como parte dos requisitos para obtenção do título de Doutora em Ciência da Computação, aprovada em 20 de maio de 2022.

# Resumo

O câncer renal é um problema de saúde pública que afeta milhares de pessoas em todo o mundo. A segmentação precisa dos rins e dos tumores renais pode ajudar os especialistas a diagnosticar doenças e melhorar o planejamento do tratamento, o que é importante na prática clínica. No entanto, devido à heterogeneidade dos rins e dos tumores renais, a segmentação manual é um processo demorado e sujeito a variabilidade entre os especialistas. Devido a esse trabalho árduo, técnicas computacionais, como redes neurais convolucionais (*Convolutional Neural Networks* - CNNs), tornaram-se populares em tarefas de segmentação automática de rins e tumores renais. As redes tridimensionais (3D) apresentam bom desempenho em tarefas de segmentação, mas são complexas e apresentam altos custos computacionais. Assim, as redes bidimensionais são as mais usadas devido ao consumo de memória relativamente baixo, mas não exploram os recursos 3D. Portanto, nesta tese, redes 2.5D, que equilibram o consumo de memória e a complexidade do modelo, são propostas para auxiliar médicos especializados na detecção de rins e tumores renais em tomografia computadorizada (TC). Estas redes estão inseridas em um método proposto organizado em quatro etapas: (1) pré-processamento da base de imagens; (2) segmentação inicial dos rins e tumores renais usando os modelos ResUNet 2.5D e DeepLabv3+ 2.5D, respectivamente; (3) reconstrução dos tumores renais usando operação binária; e (4) redução de falsos positivos usando técnicas de processamento de imagens. O método proposto foi avaliado em 210 TCs da base de imagens KiTS19. Na segmentação dos rins, apresentou 97,45\% de Dice, 95,05\% de Jaccard, 99,95\% de acurácia, 98,44\% de sensibilidade e 99,96\% de especificidade. Na segmentação dos tumores renais, foi obtido 84,06\% de Dice, 75,04\% de Jaccard, 99,94\% de acurácia, 88,33\% de sensibilidade e 99,95\% de especificidade. De maneira geral, os resultados fornecem fortes evidências de que o método proposto é uma ferramenta com potencial para auxiliar no diagnóstico da doença.

# Arquivos

Repositório da base de imagens: 
 - https://github.com/neheller/kits19

Link do desafio KiTS10: 
 - https://kits19.grand-challenge.org/
