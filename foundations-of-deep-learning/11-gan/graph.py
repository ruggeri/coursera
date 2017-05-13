from collections import namedtuple
import discriminator as discriminator
import generator as generator

NetworkConfiguration = namedtuple("NetworkConfiguration", [
    "num_classes",
    "num_discriminator_hidden_units",
    "num_generator_hidden_units",
    "num_x_dims",
    "num_z_dims",
])

Graph = namedtuple("Graph", [
    "discriminator",
    "generator",
])

def graph(network_configuration):
    # Build parameters
    discriminator_parameters = discriminator.parameters(
        network_configuration
    )
    generator_parameters = generator.parameters(
        network_configuration
    )

    # Build discriminator/generator
    d = discriminator.discriminator(
        network_configuration,
        discriminator_parameters,
    )
    g = generator.generator(
        network_configuration,
        generator_parameters,
        discriminator_parameters,
    )

    return Graph(
        discriminator = d,
        generator = g,
    )
