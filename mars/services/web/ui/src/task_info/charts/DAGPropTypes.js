import PropTypes from 'prop-types';

export const nodeType = PropTypes.arrayOf(
    PropTypes.shape({
        id: PropTypes.string.isRequired,
        name: PropTypes.string.isRequired,
    })
);

export const nodesStatusType = PropTypes.shape({
    id: PropTypes.shape({
        status: PropTypes.number.isRequired,
        progress: PropTypes.number.isRequired,
        subtaskCount: PropTypes.number,
    })
});

export const dependencyType = PropTypes.arrayOf(
    PropTypes.shape({
        fromNodeId: PropTypes.string.isRequired,
        toNodeId: PropTypes.string.isRequired,
        linkType: PropTypes.number,
    })
);

export const dagType = PropTypes.shape({
    margin: PropTypes.oneOfType([
        PropTypes.number,
        PropTypes.string,
    ]),
    padding: PropTypes.oneOfType([
        PropTypes.number,
        PropTypes.string,
    ]),
    width: PropTypes.oneOfType([
        PropTypes.number,
        PropTypes.string,
    ]),
    height: PropTypes.oneOfType([
        PropTypes.number,
        PropTypes.string,
    ]),
});
