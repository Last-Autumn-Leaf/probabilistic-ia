import sys
import problematique.run_model as model
import problematique.visualize as viz

arg2Func={'rnn': model.RNN,
          'bayes':model.Bayes,
          'knn':model.KNN,
          '2d':viz.graphs2d,
          '1d':viz.Hist1D,
          'conf':model.plotALlConfusionMatrixes}

def main(argument) :
    if argument =='main.py':
        return
    arg2Func[argument]()
    viz.plt.show()


if __name__=='__main__':
    for argument in sys.argv :
        main(argument.lower())


