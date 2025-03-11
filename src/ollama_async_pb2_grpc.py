# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""

import grpc

import ollama_async_pb2 as ollama__async__pb2

GRPC_GENERATED_VERSION = '1.70.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + ' but the generated code in ollama_async_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class OllamaServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RunModel = channel.unary_unary(
                '/ollama.OllamaService/RunModel',
                request_serializer=ollama__async__pb2.ModelRequest.SerializeToString,
                response_deserializer=ollama__async__pb2.ModelResponse.FromString,
                _registered_method=True)
        self.CreateSession = channel.unary_unary(
                '/ollama.OllamaService/CreateSession',
                request_serializer=ollama__async__pb2.SessionRequest.SerializeToString,
                response_deserializer=ollama__async__pb2.SessionResponse.FromString,
                _registered_method=True)
        self.ChatMessage = channel.unary_unary(
                '/ollama.OllamaService/ChatMessage',
                request_serializer=ollama__async__pb2.ChatRequest.SerializeToString,
                response_deserializer=ollama__async__pb2.ModelResponse.FromString,
                _registered_method=True)
        self.StreamChat = channel.unary_stream(
                '/ollama.OllamaService/StreamChat',
                request_serializer=ollama__async__pb2.ChatRequest.SerializeToString,
                response_deserializer=ollama__async__pb2.ModelResponse.FromString,
                _registered_method=True)


class OllamaServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RunModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateSession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ChatMessage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamChat(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_OllamaServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RunModel': grpc.unary_unary_rpc_method_handler(
                    servicer.RunModel,
                    request_deserializer=ollama__async__pb2.ModelRequest.FromString,
                    response_serializer=ollama__async__pb2.ModelResponse.SerializeToString,
            ),
            'CreateSession': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateSession,
                    request_deserializer=ollama__async__pb2.SessionRequest.FromString,
                    response_serializer=ollama__async__pb2.SessionResponse.SerializeToString,
            ),
            'ChatMessage': grpc.unary_unary_rpc_method_handler(
                    servicer.ChatMessage,
                    request_deserializer=ollama__async__pb2.ChatRequest.FromString,
                    response_serializer=ollama__async__pb2.ModelResponse.SerializeToString,
            ),
            'StreamChat': grpc.unary_stream_rpc_method_handler(
                    servicer.StreamChat,
                    request_deserializer=ollama__async__pb2.ChatRequest.FromString,
                    response_serializer=ollama__async__pb2.ModelResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ollama.OllamaService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('ollama.OllamaService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class OllamaService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RunModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/ollama.OllamaService/RunModel',
            ollama__async__pb2.ModelRequest.SerializeToString,
            ollama__async__pb2.ModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def CreateSession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/ollama.OllamaService/CreateSession',
            ollama__async__pb2.SessionRequest.SerializeToString,
            ollama__async__pb2.SessionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ChatMessage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/ollama.OllamaService/ChatMessage',
            ollama__async__pb2.ChatRequest.SerializeToString,
            ollama__async__pb2.ModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def StreamChat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(
            request,
            target,
            '/ollama.OllamaService/StreamChat',
            ollama__async__pb2.ChatRequest.SerializeToString,
            ollama__async__pb2.ModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
