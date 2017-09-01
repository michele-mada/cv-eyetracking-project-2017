from .mappers import neural_mapper, poly_mapper, fuzzy_mapper


mapper_implementations = {
    "neural": neural_mapper.NeuralMapper,
    "poly_quad": poly_mapper.PolyQuadMapper,
    "poly_lin": poly_mapper.PolyLinMapper,
    "fuzzy": fuzzy_mapper.FuzzyMapper,
}